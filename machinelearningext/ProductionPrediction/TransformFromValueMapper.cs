// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Converts a ValueMapper into a IDataTransform.
    /// Similar to a scorer but in a more explicit way.
    /// </summary>
    public class TransformFromValueMapper : IDataTransform, IValueMapper
    {
        #region members

        readonly IDataTransform _transform;
        readonly IDataView _source;
        readonly IHostEnvironment _host;
        readonly IValueMapper _mapper;
        readonly string _inputColumn;
        readonly string _outputColumn;
        readonly Schema _schema;

        #endregion

        #region constructors, transform API

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="mapper">IValueMapper</param>
        /// <param name="source">source to replace</param>
        /// <param name="inputColumn">name of the input column (the last one sharing the same type)</param>
        /// <param name="outputColumn">name of the output column</param>
        public TransformFromValueMapper(IHostEnvironment env, IValueMapper mapper, IDataView source,
                                        string inputColumn, string outputColumn = "output")
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(source);
            _host = env;

            if (string.IsNullOrEmpty(inputColumn))
            {
                var inputType = mapper.InputType;
                for (int i = source.Schema.ColumnCount - 1; i >= 0; --i)
                {
                    var ty = source.Schema.GetColumnType(i);
                    if (ty.SameSizeAndItemType(inputType))
                    {
                        inputColumn = source.Schema.GetColumnName(i);
                        break;
                    }
                }
            }

            _source = source;
            _mapper = mapper;
            int index;
            if (!_source.Schema.TryGetColumnIndex(inputColumn, out index))
                throw env.Except("Unable to find column '{0}' in input schema.", inputColumn);
            _inputColumn = inputColumn;
            if (_source.Schema.TryGetColumnIndex(outputColumn, out index))
                throw env.Except("Column '{0}' already present in input schema.", outputColumn);
            _outputColumn = outputColumn;
            _schema = Schema.Create(new ExtendedSchema(source.Schema, new[] { outputColumn }, new[] { mapper.OutputType }));
            _transform = CreateMemoryTransform();
        }

        public ColumnType InputType { get { return _mapper.InputType; } }
        public ColumnType OutputType { get { return _mapper.OutputType; } }
        public string InputName { get { return _inputColumn; } }
        public string OutputName { get { return _outputColumn; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>() { return _mapper.GetMapper<TSrc, TDst>(); }
        public IDataView Source { get { return _source; } }
        public bool CanShuffle { get { return _source.CanShuffle; } }
        public long? GetRowCount(bool lazy = true) { return _source.GetRowCount(lazy); }
        public Schema Schema { get { return _schema; } }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return _transform.GetRowCursor(needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _transform.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            throw _host.ExceptNotSupp("Not meant to be serialized. You need to serialize whatever it takes to instantiate it.");
        }

        #endregion

        #region Cast

        IDataTransform CreateMemoryTransform()
        {
            if (InputType.IsVector)
            {
                switch (InputType.AsVector.ItemType.RawKind)
                {
                    case DataKind.R4:
                        return CreateMemoryTransformIn<VBuffer<float>>();
                    default:
                        throw _host.ExceptNotImpl("Input Type '{0}' is not handled yet.", InputType.AsVector.ItemType.RawKind);
                }
            }
            else
            {
                switch (InputType.RawKind)
                {
                    case DataKind.R4:
                        return CreateMemoryTransformIn<float>();
                    default:
                        throw _host.ExceptNotImpl("Input Type '{0}' is not handled yet.", InputType.RawKind);
                }
            }
        }

        IDataTransform CreateMemoryTransformIn<TSrc>()
        {
            if (OutputType.IsVector)
            {
                switch (OutputType.AsVector.ItemType.RawKind)
                {
                    case DataKind.U4:
                        return CreateMemoryTransformInOut<TSrc, VBuffer<uint>>();
                    case DataKind.R4:
                        return CreateMemoryTransformInOut<TSrc, VBuffer<float>>();
                    default:
                        throw _host.ExceptNotImpl("Output Type '{0}' is not handled yet.", OutputType.AsVector.ItemType.RawKind);
                }
            }
            else
            {
                switch (OutputType.RawKind)
                {
                    case DataKind.U4:
                        return CreateMemoryTransformInOut<TSrc, uint>();
                    case DataKind.R4:
                        return CreateMemoryTransformInOut<TSrc, float>();
                    default:
                        throw _host.ExceptNotImpl("Output Type '{0}' is not handled yet.", OutputType.RawKind);
                }
            }
        }

        IDataTransform CreateMemoryTransformInOut<TSrc, TDst>()
        {
            return new MemoryTransform<TSrc, TDst>(_host, this);
        }

        #endregion

        #region memory transform

        class MemoryTransform<TSrc, TDst> : IDataTransform
        {
            readonly IHostEnvironment _host;
            readonly TransformFromValueMapper _parent;

            public MemoryTransform(IHostEnvironment env, TransformFromValueMapper parent)
            {
                _host = env;
                _parent = parent;
            }

            public TransformFromValueMapper Parent { get { return _parent; } }
            public ColumnType InputType { get { return _parent.InputType; } }
            public ColumnType OutputType { get { return _parent.OutputType; } }
            public ValueMapper<TTSrc, TTDst> GetMapper<TTSrc, TTDst>() { return _parent.GetMapper<TTSrc, TTDst>(); }
            public IDataView Source { get { return _parent.Source; } }
            public bool CanShuffle { get { return _parent.CanShuffle; } }
            public long? GetRowCount(bool lazy = true) { return _parent.GetRowCount(lazy); }
            public Schema Schema { get { return _parent.Schema; } }
            public void Save(ModelSaveContext ctx) { throw Contracts.ExceptNotSupp("Not meant to be serialized. You need to serialize whatever it takes to instantiate it."); }

            /// <summary>
            /// When the last column is requested, we also need the column used to compute it.
            /// This function ensures that this column is requested when the last one is.
            /// </summary>
            bool PredicatePropagation(int col, int index, Func<int, bool> predicate)
            {
                if (predicate(col))
                    return true;
                if (col == index)
                    return predicate(_parent.Source.Schema.ColumnCount);
                return predicate(col);
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                int index;
                if (!Source.Schema.TryGetColumnIndex(_parent.InputName, out index))
                    throw _host.Except("Unable to find column '{0}' in input schema.", _parent.InputName);
                if (predicate(index))
                {
                    var cursor = Source.GetRowCursor(i => PredicatePropagation(i, index, predicate), rand);
                    return new MemoryCursor<TSrc, TDst>(this, cursor, index);
                }
                else
                    // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                    return new SameCursor(Source.GetRowCursor(predicate, rand), Schema);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                int index;
                if (!Source.Schema.TryGetColumnIndex(_parent.InputName, out index))
                    throw _host.Except("Unable to find column '{0}' in input schema.", _parent.InputName);
                if (predicate(index))
                {
                    var cursors = Source.GetRowCursorSet(out consolidator, i => PredicatePropagation(i, index, predicate), n, rand);
                    return cursors.Select(c => new MemoryCursor<TSrc, TDst>(this, c, index)).ToArray();
                }
                else
                    // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                    return Source.GetRowCursorSet(out consolidator, predicate, n, rand)
                                 .Select(c => new SameCursor(c, Schema))
                                 .ToArray();
            }
        }

        #endregion

        #region cursor

        class MemoryCursor<TSrc, TDst> : IRowCursor
        {
            readonly MemoryTransform<TSrc, TDst> _view;
            readonly IRowCursor _inputCursor;
            readonly int _inputCol;

            public MemoryCursor(MemoryTransform<TSrc, TDst> view, IRowCursor cursor, int inputCol)
            {
                _view = view;
                _inputCursor = cursor;
                _inputCol = inputCol;
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool IsColumnActive(int col)
            {
                // The column is active if is active in the input view or if it the new vector with the polynomial features.
                return col >= _inputCursor.Schema.ColumnCount || _inputCursor.IsColumnActive(col);
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                // We do not change the ID (row to row transform).
                var getId = _inputCursor.GetIdGetter();
                return (ref UInt128 pos) =>
                {
                    getId(ref pos);
                };
            }

            public CursorState State { get { return _inputCursor.State; } } // No change.
            public long Batch { get { return _inputCursor.Batch; } }        // No change.
            public long Position { get { return _inputCursor.Position; } }  // No change.
            public Schema Schema { get { return _view.Schema; } }          // No change.

            void IDisposable.Dispose()
            {
                _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public bool MoveMany(long count)
            {
                return _inputCursor.MoveMany(count);
            }

            public bool MoveNext()
            {
                return _inputCursor.MoveNext();
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                // If the column is part of the input view.
                if (col < _inputCursor.Schema.ColumnCount)
                    return _inputCursor.GetGetter<TValue>(col);
                // If it is the added column.
                else if (col == _inputCursor.Schema.ColumnCount)
                    return GetGetterMapper() as ValueGetter<TValue>;
                // Otherwise, it is an error.
                else
                    throw Contracts.Except("Unexpected columns {0} > {1}.", col, _inputCursor.Schema.ColumnCount);
            }

            ValueGetter<TDst> GetGetterMapper()
            {
                var mapper = _view.Parent.GetMapper<TSrc, TDst>();
                var getter = _inputCursor.GetGetter<TSrc>(_inputCol);
                TSrc input = default(TSrc);
                return (ref TDst output) =>
                {
                    getter(ref input);
                    mapper(in input, ref output);
                };
            }
        }

        #endregion
    }
}
