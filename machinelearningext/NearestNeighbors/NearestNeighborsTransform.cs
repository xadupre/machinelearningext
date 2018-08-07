// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;


// The following files makes the object visible to maml.
// This way, it can be added to any pipeline.
using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using NearestNeighborsTransform = Scikit.ML.NearestNeighbors.NearestNeighborsTransform;

[assembly: LoadableClass(NearestNeighborsTransform.Summary, typeof(NearestNeighborsTransform),
    typeof(NearestNeighborsTransform.Arguments), typeof(SignatureDataTransform),
    NearestNeighborsTransform.LongName, NearestNeighborsTransform.LoaderSignature,
    NearestNeighborsTransform.ShortName)]

[assembly: LoadableClass(NearestNeighborsTransform.Summary, typeof(NearestNeighborsTransform),
    null, typeof(SignatureLoadDataTransform), NearestNeighborsTransform.LongName,
    NearestNeighborsTransform.LoaderSignature, NearestNeighborsTransform.ShortName)]


namespace Scikit.ML.NearestNeighbors
{
    public class NearestNeighborsTransform : IDataTransform
    {
        /// <summary>
        /// A unique signature.
        /// </summary>
        public const string LoaderSignature = "NearNeighborsTransform";  // Not more than 24 letters.
        public const string Summary = "Retrieve the closest neighbors among a set of points.";
        public const string RegistrationName = LoaderSignature;
        public const string LongName = "Nearest Neighbors Transform";
        public const string ShortName = "knntr";

        /// <summary>
        /// Identify the object for dynamic instantiation.
        /// This is also used to track versionning when serializing and deserializing.
        /// </summary>
        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NEARNEST",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments : NearestNeighborsArguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Feature column", ShortName = "col")]
            public string column = "Features";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Distance columns (output)", ShortName = "dist")]
            public string distColumn = "Distances";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Id of the neighbors (output)", ShortName = "idn")]
            public string idNeighborsColumn = "idNeighbors";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Label (unused) in this transform but could be leveraged later.", ShortName = "l")]
            public string labelColumn = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weights columns.", ShortName = "colw")]
            public string weightColumn = null;

            public override void Write(ModelSaveContext ctx, IHost host)
            {
                base.Write(ctx, host);
                ctx.Writer.Write(column);
                ctx.Writer.Write(distColumn);
                ctx.Writer.Write(idNeighborsColumn);
                ctx.Writer.Write(labelColumn == null ? "" : labelColumn);
                ctx.Writer.Write(weightColumn == null ? "" : weightColumn);
            }

            public override void Read(ModelLoadContext ctx, IHost host)
            {
                base.Read(ctx, host);
                column = ctx.Reader.ReadString();
                distColumn = ctx.Reader.ReadString();
                idNeighborsColumn = ctx.Reader.ReadString();
                labelColumn = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(labelColumn))
                    labelColumn = null;
                weightColumn = ctx.Reader.ReadString();
                if (string.IsNullOrEmpty(weightColumn))
                    weightColumn = null;
            }

            public override void PostProcess()
            {
                base.PostProcess();
            }
        }

        IDataView _input;
        Arguments _args;
        IHost _host;
        ISchema _extendedSchema;
        NearestNeighborsTrees _trees;
        object _lock;

        public IDataView Source { get { return _input; } }
        public NearestNeighborsTrees Trees { get { return _trees; } }

        public NearestNeighborsTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            args.PostProcess();
            _host.CheckValue(args.column, "column");

            _input = input;
            _trees = null;
            _args = args;
            _lock = new object();
            _extendedSchema = ComputeExtendedSchema();
        }

        public static NearestNeighborsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NearestNeighborsTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.Writer.Write((byte)(_trees != null ? 1 : 0));
            if (_trees != null)
                // If _trees is null, this means the pipeline was never run once.
                _trees.Save(ctx);
            _extendedSchema = ComputeExtendedSchema();
        }

        private NearestNeighborsTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _lock = new object();
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            bool run = ctx.Reader.ReadByte() == 1;
            _trees = run ? new NearestNeighborsTrees(host, ctx) : null;
            _extendedSchema = ComputeExtendedSchema();
        }

        ISchema ComputeExtendedSchema()
        {
            return new ExtendedSchema(_input.Schema, new string[] { _args.distColumn, _args.idNeighborsColumn },
                                       new ColumnType[] { new VectorType(NumberType.R4, _args.k),
                                       new VectorType(NumberType.I8, _args.k) });
        }

        public ISchema Schema { get { return _extendedSchema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount(lazy);
        }

        /// <summary>
        /// When the last column is requested, we also need the column used to compute it.
        /// This function ensures that this column is requested when the last one is.
        /// </summary>
        bool PredicatePropagation(int col, int featureIndex, Func<int, bool> predicate)
        {
            if (predicate(col))
                return true;
            if (col == featureIndex)
                return predicate(Source.Schema.ColumnCount);
            return predicate(col);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            ComputeNearestNeighbors();
            _host.AssertValue(_input, "_input");

            if (predicate(_input.Schema.ColumnCount))
            {
                int featureIndex;
                if (!_input.Schema.TryGetColumnIndex(_args.column, out featureIndex))
                    throw _host.Except("Unable to find column '{0}'.", _args.column);
                return new NearestNeighborsCursor(_input.GetRowCursor(i => PredicatePropagation(i, featureIndex, predicate), rand), this, predicate, featureIndex);
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return new SameCursor(_input.GetRowCursor(predicate, rand), Schema);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            ComputeNearestNeighbors();
            _host.AssertValue(_input, "_input");

            if (predicate(_input.Schema.ColumnCount))
            {
                int featureIndex;
                if (!_input.Schema.TryGetColumnIndex(_args.column, out featureIndex))
                    throw _host.Except("Unable to find column '{0}'.", _args.column);

                var res = _input.GetRowCursorSet(out consolidator, predicate, n, rand)
                                .Select(c => new NearestNeighborsCursor(c, this, i => PredicatePropagation(i, featureIndex, predicate), featureIndex)).ToArray();
                consolidator = new Consolidator();
                return res;
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return _input.GetRowCursorSet(out consolidator, predicate, n, rand)
                             .Select(c => new SameCursor(c, Schema))
                             .ToArray();
        }

        private sealed class Consolidator : IRowCursorConsolidator
        {
            private const int _batchShift = 6;
            private const int _batchSize = 1 << _batchShift;
            public IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs)
            {
                return DataViewUtils.ConsolidateGeneric(provider, inputs, _batchSize);
            }
        }

        int GetColumnIndex(IExceptionContext ch, string name)
        {
            if (string.IsNullOrEmpty(name))
                return -1;
            int index;
            if (!_input.Schema.TryGetColumnIndex(name, out index))
                throw ch.Except("Unable to find column '{0}'.", name);
            return index;
        }

        void ComputeNearestNeighbors()
        {
            lock (_lock)
            {
                if (_trees != null)
                    return;

                using (var ch = _host.Start("Build k-d tree"))
                {
                    ch.Info("ComputeNearestNeighbors: build a k-d tree.");
                    int featureIndex, labelIndex, idIndex, weightIndex;
                    if (!_input.Schema.TryGetColumnIndex(_args.column, out featureIndex))
                        throw ch.Except("Unable to find column '{0}'.", _args.column);
                    labelIndex = GetColumnIndex(ch, _args.labelColumn);
                    weightIndex = GetColumnIndex(ch, _args.weightColumn);
                    idIndex = GetColumnIndex(ch, _args.colId);

                    Dictionary<long, Tuple<long, float>> merged;
                    _trees = NearestNeighborsBuilder.NearestNeighborsBuild<long>(ch, _input, featureIndex, labelIndex,
                                        idIndex, weightIndex, out merged, _args);
                    ch.Info("Done. Tree size: {0} points.", _trees.Count());
                    ch.Done();
                }
            }
        }

        #region Cursor

        public class NearestNeighborsCursor : IRowCursor
        {
            readonly IRowCursor _inputCursor;
            readonly NearestNeighborsTransform _parent;
            readonly ValueGetter<VBuffer<float>> _getterFeatures;
            readonly NearestNeighborsTrees _trees;
            readonly int _k;

            VBuffer<float> _tempFeatures;
            VBuffer<float> _distance;
            VBuffer<DvInt8> _idn;

            public NearestNeighborsCursor(IRowCursor cursor, NearestNeighborsTransform parent, Func<int, bool> predicate, int colFeatures)
            {
                _inputCursor = cursor;
                _parent = parent;
                _trees = parent._trees;
                _k = parent._args.k;
                _getterFeatures = _inputCursor.GetGetter<VBuffer<float>>(colFeatures);
                _tempFeatures = new VBuffer<float>();
                _distance = new VBuffer<float>(_k, new float[_k]);
                _idn = new VBuffer<DvInt8>(_k, new DvInt8[_k]);
            }

            public ICursor GetRootCursor()
            {
                return this;
            }

            public bool IsColumnActive(int col)
            {
                return col >= _inputCursor.Schema.ColumnCount || _inputCursor.IsColumnActive(col);
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
            public ISchema Schema { get { return _parent.Schema; } }

            void IDisposable.Dispose()
            {
                _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public bool MoveMany(long count)
            {
                var res = _inputCursor.MoveMany(count);
                if (!res)
                    return res;
                RetrieveNeighbors();
                return true;
            }

            public bool MoveNext()
            {
                var res = _inputCursor.MoveNext();
                if (!res)
                    return res;
                RetrieveNeighbors();
                return true;
            }

            void RetrieveNeighbors()
            {
                _getterFeatures(ref _tempFeatures);
                var res = _trees.NearestNNeighbors(_tempFeatures, _k);
                if (res.Length > _distance.Length || res.Length > _distance.Count ||
                    res.Length > _distance.Values.Length || _distance.Values == null)
                {
                    _distance = new VBuffer<float>(res.Length, new float[res.Length]);
                    _idn = new VBuffer<DvInt8>(res.Length, new DvInt8[res.Length]);
                }
                else if (res.Length > _distance.Count)
                {
                    _distance = new VBuffer<float>(res.Length, _distance.Values);
                    _idn = new VBuffer<DvInt8>(res.Length, _idn.Values);
                }
                int pos = 0;
                foreach (var pair in res.OrderBy(c => c.Key))
                {
                    _distance.Values[pos] = pair.Key;
                    _idn.Values[pos++] = pair.Value;
                }
                Contracts.Assert(_distance.IsDense);
                Contracts.Assert(_idn.IsDense);
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                ValueGetter<TValue> res;
                if (col < _inputCursor.Schema.ColumnCount)
                    res = _inputCursor.GetGetter<TValue>(col);
                else if (col == _inputCursor.Schema.ColumnCount)
                    res = GetGetterDistance(col) as ValueGetter<TValue>;
                else if (col == _inputCursor.Schema.ColumnCount + 1)
                    res = GetGetterIdNeighbors(col) as ValueGetter<TValue>;
                else
                    throw Contracts.Except("Unexpected column position:{0}.", col);
#if(DEBUG)
                if (res == null)
                    throw _parent._host.Except("Unable to retrieve a getter for col={0} type={1} schema={2}", col, typeof(TValue), SchemaHelper.ToString(Schema));
#endif
                return res;
            }

            ValueGetter<VBuffer<float>> GetGetterDistance(int col)
            {
                if (col == _inputCursor.Schema.ColumnCount)
                    return (ref VBuffer<float> distance) =>
                    {
                        distance = new VBuffer<float>(_distance.Count, _distance.Values);
                    };
                else
                    throw Contracts.Except("Unexpected column for distance (position:{0})", col);
            }

            ValueGetter<VBuffer<DvInt8>> GetGetterIdNeighbors(int col)
            {
                if (col == _inputCursor.Schema.ColumnCount + 1)
                    return (ref VBuffer<DvInt8> distance) =>
                    {
                        distance = new VBuffer<DvInt8>(_idn.Count, _idn.Values);
                    };
                else
                    throw Contracts.Except("Unexpected column for neighbors ids (position:{0})", col);
            }
        }

        #endregion
    }
}

