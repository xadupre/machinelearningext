// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using ScalerTransform = Scikit.ML.FeaturesTransforms.ScalerTransform;

[assembly: LoadableClass(ScalerTransform.Summary, typeof(ScalerTransform),
    typeof(ScalerTransform.Arguments), typeof(SignatureDataTransform),
    ScalerTransform.LongName, ScalerTransform.LoaderSignature, ScalerTransform.ShortName)]

[assembly: LoadableClass(ScalerTransform.Summary, typeof(ScalerTransform),
    null, typeof(SignatureLoadDataTransform),
    ScalerTransform.LongName, ScalerTransform.LoaderSignature, ScalerTransform.ShortName)]


namespace Scikit.ML.FeaturesTransforms
{
    /// <summary>
    /// Normalizes columns with various stategies.
    /// </summary>
    public class ScalerTransform : IDataTransform, ITrainableTransform
    {
        public const string LoaderSignature = "ScalerTransform";  // Not more than 24 letters.
        public const string Summary = "Rescales a column (only float).";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "Scaler";
        public const string LongName = "Scaler Transform";

        /// <summary>
        /// Identify the object for dynamic instantiation.
        /// This is also used to track versionning when serializing and deserializing.
        /// </summary>
        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SCALETNS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ScalerTransform).Assembly.FullName);
        }

        public enum ScalerStrategy
        {
            meanVar = 0,
            minMax = 1
        }

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments
        {
            [Argument(ArgumentType.MultipleUnique, HelpText = "Columns to normalize.", ShortName = "col")]
            public Column1x1[] columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scaling strategy.", ShortName = "scale")]
            public ScalerStrategy scaling = ScalerStrategy.meanVar;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(Column1x1.ArrayToLine(columns));
                ctx.Writer.Write((int)scaling);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                string sr = ctx.Reader.ReadString();
                columns = Column1x1.ParseMulti(sr);
                scaling = (ScalerStrategy)ctx.Reader.ReadInt32();
            }

            public void PostProcess()
            {
            }
        }

        IDataView _input;
        Arguments _args;
        Dictionary<string, List<ColumnStatObs>> _scalingStat;
        Dictionary<int, ScalingFactor> _scalingFactors;
        Dictionary<int, int> _revIndex;
        IHost _host;
        Schema _extendedSchema;
        object _lock;

        public IDataView Source { get { return _input; } }

        public ScalerTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, "args");
            args.PostProcess();
            _host.CheckValue(args.columns, "columns");

            _input = input;
            _args = args;
            _lock = new object();
            _scalingStat = null;
            _scalingFactors = null;
            _revIndex = null;
            _extendedSchema = ComputeExtendedSchema();
        }

        public static ScalerTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ScalerTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
            ctx.Writer.Write(_scalingStat == null ? 0 : _scalingStat.Count);
            if (_scalingFactors != null)
            {
                foreach (var pair in _scalingStat)
                {
                    ctx.Writer.Write(pair.Key);
                    ctx.Writer.Write(pair.Value.Count);
                    foreach (var val in pair.Value)
                        val.Write(ctx);
                }
            }
        }

        private ScalerTransform(IHost host, ModelLoadContext ctx, IDataView input)
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
            int nbStat = ctx.Reader.ReadInt32();
            _extendedSchema = ComputeExtendedSchema();
            if (nbStat == 0)
            {
                _scalingFactors = null;
                _scalingStat = null;
                _revIndex = null;
            }
            else
            {
                _scalingStat = new Dictionary<string, List<ColumnStatObs>>();
                for (int i = 0; i < nbStat; ++i)
                {
                    string key = ctx.Reader.ReadString();
                    int nb = ctx.Reader.ReadInt32();
                    var li = new List<ColumnStatObs>();
                    for (int k = 0; k < nb; ++k)
                        li.Add(new ColumnStatObs(ctx));
                    _scalingStat[key] = li;
                }
                _scalingFactors = GetScalingParameters();
                _revIndex = ComputeRevIndex();
            }
        }

        Schema ComputeExtendedSchema()
        {
            int index;
            Func<string, ColumnType> getType = (string col) =>
            {
                var schema = _input.Schema;
                if (!schema.TryGetColumnIndex(col, out index))
                    throw _host.Except("Unable to find column '{0}'.", col);
                return schema.GetColumnType(index);
            };
            var iterCols = _args.columns.Where(c => c.Name != c.Source);
            return iterCols.Any()
                        ? Schema.Create(new ExtendedSchema(_input.Schema,
                                                iterCols.Select(c => c.Name).ToArray(),
                                                iterCols.Select(c => getType(c.Source)).ToArray()))
                        : _input.Schema;
        }

        public Schema Schema { get { return _extendedSchema; } }

        public bool CanShuffle { get { return _input.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public long? GetRowCount()
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount();
        }

        /// <summary>
        /// When the last column is requested, we also need the column used to compute it.
        /// This function ensures that this column is requested when the last one is.
        /// </summary>
        bool PredicatePropagation(int col, Func<int, bool> predicate)
        {
            if (predicate(col))
                return true;
            if (_revIndex.ContainsKey(col))
                return predicate(_revIndex[col]);
            return predicate(col);
        }

        public RowCursor GetRowCursor(Func<int, bool> predicate, Random rand = null)
        {
            ComputeStatistics();
            _host.AssertValue(_input, "_input");
            var cursor = _input.GetRowCursor(i => PredicatePropagation(i, predicate), rand);
            return new ScalerCursor(cursor, this, i => PredicatePropagation(i, predicate));
        }

        public RowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, Random rand = null)
        {
            ComputeStatistics();
            _host.AssertValue(_input, "_input");
            var cursors = _input.GetRowCursorSet(out consolidator, i => PredicatePropagation(i, predicate), n, rand);
            var res = cursors.Select(c => new ScalerCursor(c, this, i => PredicatePropagation(i, predicate))).ToArray();
            consolidator = new Consolidator();
            return res;
        }

        private sealed class Consolidator : IRowCursorConsolidator
        {
            private const int _batchShift = 6;
            private const int _batchSize = 1 << _batchShift;
            public RowCursor CreateCursor(IChannelProvider provider, RowCursor[] inputs)
            {
                return DataViewUtils.ConsolidateGeneric(provider, inputs, _batchSize);
            }
        }

        public void Estimate()
        {
            ComputeStatistics();
        }

        void ComputeStatistics()
        {
            lock (_lock)
            {
                if (_scalingStat == null)
                {
                    using (var ch = _host.Start("ScalerTransform"))
                    {
                        var sch = _input.Schema;
                        var indexesCol = new List<int>();

                        var textCols = _args.columns.Select(c => c.Source).ToArray();
                        _scalingStat = new Dictionary<string, List<ColumnStatObs>>();

                        for (int i = 0; i < textCols.Length; ++i)
                        {
                            int index;
                            if (!sch.TryGetColumnIndex(textCols[i], out index))
                                throw ch.Except("Unable to find column '{0}' in '{1}'", textCols[i], SchemaHelper.ToString(sch));
                            var ty = sch.GetColumnType(index);
                            if (!(ty == NumberType.R4 || ty == NumberType.U4 || ty == TextType.Instance || ty == BoolType.Instance ||
                                (ty.IsKey() && ty.AsKey().RawKind() == DataKind.U4) || (ty.IsVector() && ty.AsVector().ItemType() == NumberType.R4)))
                                throw ch.Except("Only a float or a vector of floats or a uint or a text or a bool is allowed for column {0} (schema={1}).", _args.columns[i], SchemaHelper.ToString(sch));
                            indexesCol.Add(index);
                        }

                        // Computation
                        var required = new HashSet<int>(indexesCol);
                        var requiredIndexes = required.OrderBy(c => c).ToArray();
                        using (var cur = _input.GetRowCursor(i => required.Contains(i)))
                        {
                            bool[] isText = requiredIndexes.Select(c => sch.GetColumnType(c) == TextType.Instance).ToArray();
                            bool[] isBool = requiredIndexes.Select(c => sch.GetColumnType(c) == BoolType.Instance).ToArray();
                            bool[] isFloat = requiredIndexes.Select(c => sch.GetColumnType(c) == NumberType.R4).ToArray();
                            bool[] isUint = requiredIndexes.Select(c => sch.GetColumnType(c) == NumberType.U4 || sch.GetColumnType(c).RawKind() == DataKind.U4).ToArray();
                            ValueGetter<bool>[] boolGetters = requiredIndexes.Select(i => sch.GetColumnType(i) == BoolType.Instance || sch.GetColumnType(i).RawKind() == DataKind.BL ? cur.GetGetter<bool>(i) : null).ToArray();
                            ValueGetter<uint>[] uintGetters = requiredIndexes.Select(i => sch.GetColumnType(i) == NumberType.U4 || sch.GetColumnType(i).RawKind() == DataKind.U4 ? cur.GetGetter<uint>(i) : null).ToArray();
                            ValueGetter<ReadOnlyMemory<char>>[] textGetters = requiredIndexes.Select(i => sch.GetColumnType(i) == TextType.Instance ? cur.GetGetter<ReadOnlyMemory<char>>(i) : null).ToArray();
                            ValueGetter<float>[] floatGetters = requiredIndexes.Select(i => sch.GetColumnType(i) == NumberType.R4 ? cur.GetGetter<float>(i) : null).ToArray();
                            ValueGetter<VBuffer<float>>[] vectorGetters = requiredIndexes.Select(i => sch.GetColumnType(i).IsVector() ? cur.GetGetter<VBuffer<float>>(i) : null).ToArray();

                            var schema = _input.Schema;
                            for (int i = 0; i < schema.ColumnCount; ++i)
                            {
                                string name = schema.GetColumnName(i);
                                if (!required.Contains(i))
                                    continue;
                                _scalingStat[name] = new List<ColumnStatObs>();
                                var t = _scalingStat[name];
                                switch (_args.scaling)
                                {
                                    case ScalerStrategy.meanVar:
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.sum));
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.sum2));
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.nb));
                                        break;
                                    case ScalerStrategy.minMax:
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.min));
                                        t.Add(new ColumnStatObs(ColumnStatObs.StatKind.max));
                                        break;
                                    default:
                                        throw _host.ExceptNotSupp($"Unsupported scaling strategy: {_args.scaling}.");
                                }
                            }

                            float value = 0;
                            var tvalue = new ReadOnlyMemory<char>();
                            VBuffer<float> vector = new VBuffer<float>();
                            uint uvalue = 0;
                            bool bvalue = true;
                            var curschema = cur.Schema;

                            while (cur.MoveNext())
                            {
                                for (int i = 0; i < requiredIndexes.Length; ++i)
                                {
                                    string name = curschema.GetColumnName(requiredIndexes[i]);
                                    if (!_scalingStat.ContainsKey(name))
                                        continue;
                                    if (isFloat[i])
                                    {
                                        floatGetters[i](ref value);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(value);
                                    }
                                    else if (isBool[i])
                                    {
                                        boolGetters[i](ref bvalue);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(bvalue);
                                    }
                                    else if (isText[i])
                                    {
                                        textGetters[i](ref tvalue);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(tvalue.ToString());
                                    }
                                    else if (isUint[i])
                                    {
                                        uintGetters[i](ref uvalue);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(uvalue);
                                    }
                                    else
                                    {
                                        vectorGetters[i](ref vector);
                                        foreach (var t in _scalingStat[name])
                                            t.Update(vector);
                                    }
                                }
                            }
                        }

                        _scalingFactors = GetScalingParameters();
                        _revIndex = ComputeRevIndex();
                    }
                }
            }
        }

        Dictionary<int, int> ComputeRevIndex()
        {
            var revIndex = new Dictionary<int, int>();
            foreach (var pair in _scalingFactors)
                revIndex[pair.Value.columnId] = pair.Key;
            return revIndex;
        }

        public enum ScalingMethod
        {
            Affine = 0
        };

        public class ScalingFactor
        {
            public ScalingMethod scalingMethod;

            // Y = scale (X - mean)
            public int columnId;
            public VBuffer<float> mean;
            public VBuffer<float> scale;

            public ScalingFactor(int colid, ScalingMethod method, VBuffer<float> mean, VBuffer<float> scale)
            {
                scalingMethod = method;
                columnId = colid;
                this.mean = mean;
                this.scale = scale;
            }

            public ScalingFactor(IHost host, int colid, ScalerStrategy strategy, List<ColumnStatObs> obs)
            {
                columnId = colid;
                switch (strategy)
                {
                    case ScalerStrategy.meanVar:
                        scalingMethod = ComputeMeanVar(host, obs, out mean, out scale);
                        break;
                    case ScalerStrategy.minMax:
                        scalingMethod = ComputeMinMax(host, obs, out mean, out scale);
                        break;
                    default:
                        throw host.ExceptNotSupp($"Unknown scaling strategy {strategy}.");
                }
            }

            ScalingMethod ComputeMeanVar(IHost host, List<ColumnStatObs> stats,
                                         out VBuffer<float> mean, out VBuffer<float> variance)
            {
                var nb = stats.Where(c => c.kind == ColumnStatObs.StatKind.nb).ToArray();
                var sum = stats.Where(c => c.kind == ColumnStatObs.StatKind.sum).ToArray();
                var sum2 = stats.Where(c => c.kind == ColumnStatObs.StatKind.sum2).ToArray();
                if (nb.Length != 1)
                    throw host.Except("nb is null");
                if (sum.Length != 1)
                    throw host.Except("sum is null");
                if (sum2.Length != 1)
                    throw host.Except("sum2 is null");
                var dnb = nb[0].stat.DenseValues().ToArray();
                var dsum = sum[0].stat.DenseValues().ToArray();
                var dsum2 = sum2[0].stat.DenseValues().ToArray();
                if (dnb.Length != dsum.Length)
                    throw host.Except("{0} != {1}", dnb.Length, dsum.Length);
                if (dnb.Length != dsum2.Length)
                    throw host.Except("{0} != {1}", dnb.Length, dsum2.Length);
                var dmean = new float[dnb.Length];
                var dvar = new float[dnb.Length];
                for (int i = 0; i < dmean.Length; ++i)
                {
                    dmean[i] = (float)(dnb[i] == 0 ? 0 : dsum[i] / dnb[i]);
                    dvar[i] = dnb[i] == 0 ? 0 : (float)Math.Sqrt(dsum2[i] / dnb[i] - dmean[i] * dmean[i]);
                    if (dvar[i] != 0)
                        dvar[i] = 1f / dvar[i];
                }
                mean = new VBuffer<float>(dmean.Length, dmean);
                variance = new VBuffer<float>(dvar.Length, dvar);
                return ScalingMethod.Affine;
            }

            ScalingMethod ComputeMinMax(IHost host, List<ColumnStatObs> stats,
                                        out VBuffer<float> mean, out VBuffer<float> scale)
            {
                var min = stats.Where(c => c.kind == ColumnStatObs.StatKind.min).ToArray();
                var max = stats.Where(c => c.kind == ColumnStatObs.StatKind.max).ToArray();
                if (min.Length != 1)
                    throw host.Except("sum is null");
                if (max.Length != 1)
                    throw host.Except("sum2 is null");
                var dmin = min[0].stat.DenseValues().ToArray();
                var dmax = max[0].stat.DenseValues().ToArray();
                if (dmin.Length != dmax.Length)
                    throw host.Except("{0} != {1}", dmin.Length, dmax.Length);
                var dmean = new float[dmin.Length];
                var dscale = new float[dmin.Length];
                double delta;
                for (int i = 0; i < dmean.Length; ++i)
                {
                    dmean[i] = (float)(dmin[i]);
                    delta = dmax[i] - dmin[i];
                    dscale[i] = (float)(delta == 0 ? 1.0 : 1.0 / delta);
                }
                mean = new VBuffer<float>(dmean.Length, dmean);
                scale = new VBuffer<float>(dscale.Length, dscale);
                return ScalingMethod.Affine;
            }

            public void Update(ref VBuffer<float> dst)
            {
                switch (scalingMethod)
                {
                    case ScalingMethod.Affine:
                        if (dst.IsDense)
                        {
                            for (int i = 0; i < dst.Count; ++i)
                            {
                                dst.Values[i] -= mean.Values[i];
                                if (scale.Values[i] != 0f)
                                    dst.Values[i] *= scale.Values[i];
                            }
                        }
                        else
                        {
                            for (int i = 0; i < dst.Count; ++i)
                            {
                                dst.Values[i] -= mean.Values[dst.Indices[i]];
                                if (scale.Values[dst.Indices[i]] != 0f)
                                    dst.Values[i] *= scale.Values[dst.Indices[i]];
                            }
                        }
                        break;
                    default:
                        throw Contracts.ExceptNotSupp($"Unknown scaling method: {scalingMethod}.");
                }
            }
        }

        Dictionary<int, ScalingFactor> GetScalingParameters()
        {
            var res = new Dictionary<int, ScalingFactor>();
            int index, index2;
            var thisSchema = Schema;
            var schema = _input.Schema;
            for (int i = 0; i < _args.columns.Length; ++i)
            {
                if (!schema.TryGetColumnIndex(_args.columns[i].Source, out index))
                    throw _host.Except("Unable to find column '{0}'.", _args.columns[i].Source);

                string name = thisSchema.GetColumnName(index);
                var stats = _scalingStat[name];

                if (_args.columns[i].Source == _args.columns[i].Name)
                    res[index] = new ScalingFactor(_host, index, _args.scaling, stats);
                else
                {
                    if (!Schema.TryGetColumnIndex(_args.columns[i].Name, out index2))
                        throw _host.Except("Unable to find column '{0}'.", _args.columns[i].Name);
                    res[index2] = new ScalingFactor(_host, index, _args.scaling, stats);
                }
            }
            return res;
        }

        #region Cursor

        public class ScalerCursor : RowCursor
        {
            readonly RowCursor _inputCursor;
            readonly ScalerTransform _parent;
            readonly Dictionary<int, ScalingFactor> _scalingFactors;

            public ScalerCursor(RowCursor cursor, ScalerTransform parent, Func<int, bool> predicate)
            {
                _inputCursor = cursor;
                _parent = parent;
                _scalingFactors = parent._scalingFactors;
                if (_scalingFactors == null)
                    throw parent._host.ExceptValue("The transform was never trained. Predictions cannot be computed.");
            }

            public override RowCursor GetRootCursor()
            {
                return this;
            }

            public override bool IsColumnActive(int col)
            {
                return col >= _inputCursor.Schema.ColumnCount || _inputCursor.IsColumnActive(col);
            }

            public override ValueGetter<UInt128> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref UInt128 pos) =>
                {
                    getId(ref pos);
                };
            }

            public override CursorState State { get { return _inputCursor.State; } }
            public override long Batch { get { return _inputCursor.Batch; } }
            public override long Position { get { return _inputCursor.Position; } }
            public override Schema Schema { get { return _parent.Schema; } }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveMany(long count)
            {
                return _inputCursor.MoveMany(count);
            }

            public override bool MoveNext()
            {
                return _inputCursor.MoveNext();
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                var schema = _inputCursor.Schema;
                if (_scalingFactors.ContainsKey(col))
                {
                    var type = schema.GetColumnType(_scalingFactors[col].columnId);
                    if (type.IsVector())
                        return GetGetterVector(_scalingFactors[col]) as ValueGetter<TValue>;
                    else
                        return GetGetter(_scalingFactors[col]) as ValueGetter<TValue>;
                }
                else if (col < schema.ColumnCount)
                    return _inputCursor.GetGetter<TValue>(col);
                else
                    throw Contracts.Except("Unexpected columns {0}.", col);
            }

            ValueGetter<VBuffer<float>> GetGetter(ScalingFactor scales)
            {
                var getter = _inputCursor.GetGetter<float>(scales.columnId);
                float value = 0f;
                return (ref VBuffer<float> dst) =>
                {
                    getter(ref value);
                    if (1 != scales.mean.Length)
                        throw _parent._host.Except("Mismatch dimension {0} for destination != {1} for scaling vectors.", dst.Length, scales.mean.Length);
                    if (dst.Length != 1)
                        dst = new VBuffer<float>(1, new[] { value });
                    else
                        dst.Values[0] = value;
                    scales.Update(ref dst);
                };
            }

            ValueGetter<VBuffer<float>> GetGetterVector(ScalingFactor scales)
            {
                var getter = _inputCursor.GetGetter<VBuffer<float>>(scales.columnId);
                return (ref VBuffer<float> dst) =>
                {
                    getter(ref dst);
                    if (dst.Length != scales.mean.Length)
                        throw _parent._host.Except("Mismatch dimension {0} for destination != {1} for scaling vectors.", dst.Length, scales.mean.Length);
                    scales.Update(ref dst);
                };
            }
        }

        #endregion
    }
}

