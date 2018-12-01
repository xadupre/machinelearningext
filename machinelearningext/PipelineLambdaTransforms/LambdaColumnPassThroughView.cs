//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.PipelineLambdaTransforms
{
    public static class LambdaColumnHelper
    {
        /// <summary>
        /// Modifies or add another column by applying a lambda column.
        /// </summary>
        /// <typeparam name="TSrc">input type</typeparam>
        /// <typeparam name="TDst">output type</typeparam>
        /// <param name="env">environment</param>
        /// <param name="name">name to register</param>
        /// <param name="input">input view</param>
        /// <param name="src">name of the input column</param>
        /// <param name="dst">name of the output column</param>
        /// <param name="typeSrc">input column type</param>
        /// <param name="typeDst">output column type</param>
        /// <param name="mapper">mapper to apply</param>
        /// <returns>IDataView</returns>
        public static IDataView Create<TSrc, TDst>(IHostEnvironment env, string name, IDataView input,
                        string src, string dst, ColumnType typeSrc, ColumnType typeDst,
                        ValueMapper<TSrc, TDst> mapper)
        {
            return new LambdaColumnPassThroughView<TSrc, TDst>(env, name, input, src, dst, typeSrc, typeDst, mapper);
        }
    }

    /// <summary>
    /// This implement a view which runs a lambda function.
    /// The view is more simple than <see cref="LambdaColumnTransform"/>
    /// and tends to make it as a pass-through transform.
    /// </summary>
    public class LambdaColumnPassThroughView<TSrc, TDst> : IDataView
    {
        readonly IDataView _source;
        readonly string _columnSrc;
        readonly string _columnDst;
        readonly ColumnType _typeSrc;
        readonly ColumnType _typeDst;
        readonly ValueMapper<TSrc, TDst> _mapper;
        readonly Schema _newSchema;
        readonly IHost _host;
        readonly int _srcIndex;

        public IDataView SourceTags { get { return _source; } }
        protected IHost Host { get { return _host; } }

        public LambdaColumnPassThroughView(IHostEnvironment env, string name, IDataView input,
                        string src, string dst, ColumnType typeSrc, ColumnType typeDst,
                        ValueMapper<TSrc, TDst> mapper)
        {
            _host = env.Register(name);
            _source = input;
            _mapper = mapper;
            _columnDst = dst;
            _columnSrc = src;
            _typeDst = typeDst;
            _typeSrc = typeSrc;
            _newSchema = Schema.Create(new ExtendedSchema(_source.Schema, new[] { dst }, new[] { typeDst }));
            if (!_source.Schema.TryGetColumnIndex(_columnSrc, out _srcIndex))
                _host.Except("Unable to find column '{0}' in input schema.", _columnSrc);
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public Schema Schema
        {
            get { return _newSchema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        /// <summary>
        /// When the last column is requested, we also need the column used to compute it.
        /// This function ensures that this column is requested when the last one is.
        /// </summary>
        bool PredicatePropagation(int col, Func<int, bool> predicate)
        {
            if (predicate(col))
                return true;
            if (col == _srcIndex)
                return predicate(_source.Schema.ColumnCount);
            return predicate(col);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, Random rand = null)
        {
            if (predicate(_source.Schema.ColumnCount))
            {
                var cursor = _source.GetRowCursor(i => PredicatePropagation(i, predicate), rand);
                return new LambdaCursor(this, cursor);
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return new SameCursor(_source.GetRowCursor(predicate, rand), Schema);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, Random rand = null)
        {
            if (predicate(_source.Schema.ColumnCount))
            {
                var cursors = _source.GetRowCursorSet(out consolidator, i => PredicatePropagation(i, predicate), n, rand);
                return cursors.Select(c => new LambdaCursor(this, c)).ToArray();
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return _source.GetRowCursorSet(out consolidator, predicate, n, rand)
                              .Select(c => new SameCursor(c, Schema))
                              .ToArray();
        }

        public class LambdaCursor : IRowCursor
        {
            readonly LambdaColumnPassThroughView<TSrc, TDst> _view;
            readonly IRowCursor _inputCursor;

            public LambdaCursor(LambdaColumnPassThroughView<TSrc, TDst> view, IRowCursor cursor)
            {
                _view = view;
                _inputCursor = cursor;
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool IsColumnActive(int col)
            {
                if (col < _inputCursor.Schema.ColumnCount)
                    return _inputCursor.IsColumnActive(col);
                return true;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref UInt128 pos) =>
                {
                    getId(ref pos);
                };
            }

            public CursorState State { get { return _inputCursor.State; } }
            public long Batch { get { return _inputCursor.Batch; } }
            public long Position { get { return _inputCursor.Position; } }
            public Schema Schema { get { return _view.Schema; } }

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
                if (col < _view.SourceTags.Schema.ColumnCount)
                {
                    return _inputCursor.GetGetter<TValue>(col);
                }
                else if (col == _view.SourceTags.Schema.ColumnCount)
                {
                    return GetLambdaGetter() as ValueGetter<TValue>;
                }
                else
                    throw _view.Host.Except("Column index {0} does not exist.", col);
            }

            protected ValueGetter<TDst> GetLambdaGetter()
            {
                var getter = _inputCursor.GetGetter<TSrc>(_view._srcIndex);
                TSrc temp = default(TSrc);
                return (ref TDst dst) =>
                {
                    getter(ref temp);
                    _view._mapper(in temp, ref dst);
                };
            }
        }
    }
}
