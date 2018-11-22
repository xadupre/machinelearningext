// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;
using Scikit.ML.NearestNeighbors;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using OpticsOrderingTransform = Scikit.ML.Clustering.OpticsOrderingTransform;


[assembly: LoadableClass(OpticsOrderingTransform.Summary, typeof(OpticsOrderingTransform),
    typeof(OpticsOrderingTransform.Arguments), typeof(SignatureDataTransform),
    "OPTICS Ordering Transform", OpticsOrderingTransform.LoaderSignature,
    "OPTICSOrdering", "OPTICSOrd")]

[assembly: LoadableClass(OpticsOrderingTransform.Summary, typeof(OpticsOrderingTransform),
    null, typeof(SignatureLoadDataTransform), "OPTICS Ordering Transform",
    "OPTICS Ordering Transform", OpticsOrderingTransform.LoaderSignature)]


namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Transform which applies the Optics ordering algorithm.
    /// </summary>
    public class OpticsOrderingTransform : TransformBase
    {
        #region identification

        public const string LoaderSignature = "OpticsOrderingTransform";
        public const string Summary = "Orders data using OPTICS algorithm.";
        public const string RegistrationName = LoaderSignature;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "OPTORDME",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OpticsOrderingTransform).Assembly.FullName);
        }

        #endregion

        #region parameters / command line

        public class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Column which contains the features.", ShortName = "col")]
            public string features;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Radius of the sample areas. If null, the transform will give it a default value based on the data.", ShortName = "eps")]
            public double epsilon = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum number of points in the sample area to be considered a cluster.", ShortName = "mps")]
            public int minPoints = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Ordering results.", ShortName = "outc")]
            public string outOrdering = "Ordering";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reachability distance", ShortName = "outr")]
            public string outReachabilityDistance = "Reachability";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Core distance", ShortName = "outcd")]
            public string outCoreDistance = "Core";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the number generators.", ShortName = "s")]
            public int? seed = 42;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(features);
                ctx.Writer.Write(epsilon);
                ctx.Writer.Write(minPoints);
                ctx.Writer.Write(outOrdering);
                ctx.Writer.Write(outReachabilityDistance);
                ctx.Writer.Write(outCoreDistance);
                ctx.Writer.Write(seed ?? -1);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                features = ctx.Reader.ReadString();
                epsilon = ctx.Reader.ReadDouble();
                minPoints = ctx.Reader.ReadInt32();
                outOrdering = ctx.Reader.ReadString();
                outReachabilityDistance = ctx.Reader.ReadString();
                outCoreDistance = ctx.Reader.ReadString();
                int s = ctx.Reader.ReadInt32();
                seed = s < 0 ? (int?)null : s;
            }
        }

        #endregion

        #region internal members / accessors

        IDataTransform _transform;      // templated transform (not the serialized version)
        Arguments _args;                // parameters
        Schema _schema;                 // We need the schema the transform outputs.

        public override Schema Schema { get { return _schema; } }

        #endregion

        #region public constructor / serialization / load / save

        public OpticsOrderingTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, "args");

            if (args.epsilon < 0)
                Contracts.Check(false, "Parameter epsilon must be positive or null.");

            if (args.minPoints <= 0)
                Contracts.Check(false, "Parameter minPoints must be positive.");

            _args = args;
            _schema = Schema.Create(new ExtendedSchema(input.Schema, new string[] { args.outOrdering, args.outReachabilityDistance, args.outCoreDistance },
                                                       new ColumnType[] { NumberType.I8, NumberType.R4, NumberType.R4 }));
            _transform = CreateTemplatedTransform();
        }
        public static OpticsOrderingTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new OpticsOrderingTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, Host);
        }

        private OpticsOrderingTransform(IHost host, ModelLoadContext ctx, IDataView input) :
            base(host, input)
        {
            Host.CheckValue(input, "input");
            Host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, Host);
            _schema = Schema.Create(new ExtendedSchema(input.Schema, new string[] { _args.outOrdering, _args.outReachabilityDistance, _args.outCoreDistance },
                                                       new ColumnType[] { NumberType.I8, NumberType.R4, NumberType.R4 }));
            _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        public override bool CanShuffle { get { return _transform.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public override long? GetRowCount()
        {
            Host.AssertValue(Source, "_input");
            return Source.GetRowCount();
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return false;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> needCol, IRandom rand = null)
        {
            Host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(needCol, rand);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            Host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        #endregion

        #region transform own logic

        private IDataTransform CreateTemplatedTransform()
        {
            int index;
            if (!Source.Schema.TryGetColumnIndex(_args.features, out index))
                throw Host.Except("Features does not belong the input schema.");
            var type = Source.Schema.GetColumnType(index);
            if (!type.IsVector())
                throw Host.Except("Features must be a vector.");
            switch (type.AsVector().ItemType().RawKind())
            {
                case DataKind.R4:
                    return new OpticsOrderingState(Host, this, Source, _args);
                default:
                    throw Host.Except("Features must be a vector a floats.");
            }
        }

        public class OpticsOrderingState : IDataTransform
        {
            IHost _host;
            IDataView _input;
            Arguments _args;
            OpticsOrderingTransform _parent;
            OpticsOrderingResult[] _Results;              // OptictsOrderingResult: result of a point
            Dictionary<long, long> _reversedMapping;      // long: index of a point

            /// <summary>
            /// Array of cluster.
            /// </summary>
            public OpticsOrderingResult[] OpticsOrderingResults { get { return _Results; } }

            /// <summary>
            /// To retrieve the index within the ordering of one observation.
            /// </summary>
            public long GetMappedIndex(long vertexId) { return _reversedMapping[vertexId]; }

            object _lock;

            public IDataView Source { get { return _input; } }
            public Schema Schema { get { return _parent.Schema; } }

            public OpticsOrderingState(IHostEnvironment host, OpticsOrderingTransform parent, IDataView input, Arguments args)
            {
                _host = host.Register("OpticsOrderingState");
                _host.CheckValue(input, "input");
                _input = input;
                _lock = new object();
                _args = args;
                _Results = null;
                _parent = parent;
            }

            void TrainTransform()
            {
                lock (_lock)
                {
                    if (_Results != null)
                        return;

                    using (var ch = _host.Start("Starting Optics"))
                    {
                        var sw = Stopwatch.StartNew();
                        sw.Start();
                        var points = new List<IPointIdFloat>();

                        int index;
                        if (!_input.Schema.TryGetColumnIndex(_args.features, out index))
                            ch.Except("Unable to find column '{0}'", _args.features);

                        // Caching data.
                        ch.Info("Caching the data.");
                        using (var cursor = _input.GetRowCursor(i => i == index))
                        {
                            var getter = cursor.GetGetter<VBuffer<float>>(index);
                            var getterId = cursor.GetIdGetter();
                            UInt128 id = new UInt128();

                            VBuffer<float> tmp = new VBuffer<float>();

                            for (int i = 0; cursor.MoveNext(); i++)
                            {
                                getter(ref tmp);
                                getterId(ref id);
                                if (id > long.MaxValue)
                                    ch.Except("An id is outside the range for long {0}", id);
                                points.Add(new PointIdFloat((long)id, tmp.DenseValues().Select(c => (float)c)));
                            }
                        }

                        // Mapping.
                        // long: index in the ordering
                        // long: index of a point
                        var mapping = new long[points.Count];
                        var mapprev = new Dictionary<long, long>();

                        var distance = (float)_args.epsilon;
                        if (distance <= 0)
                        {
                            float mind, maxd;
                            distance = EstimateDistance(ch, points, out mind, out maxd);
                            ch.Info("epsilon (=Radius) was estimating on random couples of points: {0} in [{1}, {2}]", distance, mind, maxd);
                        }

                        Optics opticsAlgo = new Optics(points, _args.seed);
                        //Ordering
                        ch.Info("Generating OPTICS ordering for {0} points.", points.Count);
                        int nPoints = points.Count;
                        int cyclesBetweenLogging = Math.Min(1000, nPoints / 10);
                        int currentIteration = 0;

                        Action progressLogger = () =>
                        {
                            if (++currentIteration % cyclesBetweenLogging == 0)
                            {
                                ch.Info("Processing {0}/{1}", currentIteration, nPoints);
                            }
                        };

                        OpticsOrdering opticsOrdering = opticsAlgo.Ordering(
                            distance,
                            _args.minPoints,
                            seed: _args.seed,
                            onShuffle: msg => ch.Info(msg),
                            onPointProcessing: progressLogger);
                        IReadOnlyDictionary<long, long> results = opticsOrdering.orderingMapping;
                        var reachabilityDs = opticsOrdering.reachabilityDistances;
                        var coreDs = opticsOrdering.coreDistancesCache;

                        for (int i = 0; i < results.Count; ++i)
                        {
                            var p = points[i];
                            mapprev[results[i]] = i;
                            mapping[i] = results[i];
                        }
                        _reversedMapping = mapprev;

                        // Cleaning.
                        ch.Info("Cleaning.");
                        // We replace by the original labels.
                        _Results = new OpticsOrderingResult[results.Count];

                        for (int i = 0; i < results.Count; ++i)
                        {
                            long pId = points[i].id;
                            float? rd;
                            float? cd;

                            reachabilityDs.TryGetValue(pId, out rd);
                            coreDs.TryGetValue(pId, out cd);

                            _Results[i] = new OpticsOrderingResult()
                            {
                                id = results[i] != DBScan.NOISE ? results[i] : -1,
                                reachability = (float)rd.GetValueOrDefault(float.PositiveInfinity),
                                core = (float)cd.GetValueOrDefault(float.PositiveInfinity)
                            };
                        }
                        ch.Info("Ordered {0} points.", _Results.Count());
                        sw.Stop();
                        ch.Info("'OpticsOrdering' finished in {0}.", sw.Elapsed);
                    }
                }
            }

            public float EstimateDistance(IChannel ch, List<IPointIdFloat> points, out float minDistance, out float maxDistance)
            {
                ch.Info("Estimating epsilon based on the data. We pick up two random random computes the average distance.");
                var rand = _args.seed.HasValue ? new Random(_args.seed.Value) : new Random();
                var stack = new List<float>();
                float sum = 0, sum2 = 0;
                float d;
                int nb = 0;
                float ave = 0, last = 0;
                int i, j;
                while (stack.Count < 10000)
                {
                    i = rand.Next(0, points.Count - 1);
                    j = rand.Next(0, points.Count - 1);
                    if (i == j)
                        continue;
                    d = points[i].DistanceTo(points[j]);
                    sum += d;
                    sum2 += d * d;
                    stack.Add(d);
                    ++nb;
                    if (nb > 10)
                    {
                        ave = sum2 / nb;
                        if (Math.Abs(ave - last) < 1e-5)
                            break;
                    }
                    if (nb > 9)
                        last = ave;
                }
                if (stack.Count == 0)
                    throw ch.Except("The radius cannot be estimated.");
                stack.Sort();
                if (!stack.Where(c => !float.IsNaN(c)).Any())
                    throw ch.Except("All distances are NaN. Check your datasets.");
                minDistance = stack.Where(c => !float.IsNaN(c)).First();
                maxDistance = stack.Last();
                return stack[Math.Min(stack.Count - 1, Math.Max(stack.Count / 20, 2))];
            }

            public bool CanShuffle { get { return true; } }

            public long? GetRowCount()
            {
                return _input.GetRowCount();
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                TrainTransform();
                _host.AssertValue(_Results, "_Results");
                var cursor = _input.GetRowCursor(predicate, rand);
                return new OpticsOrderingCursor(this, cursor);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                TrainTransform();
                _host.AssertValue(_Results, "_Results");
                var cursors = _input.GetRowCursorSet(out consolidator, predicate, n, rand);
                return cursors.Select(c => new OpticsOrderingCursor(this, c)).ToArray();
            }

            public void Save(ModelSaveContext ctx)
            {
                throw Contracts.ExceptNotSupp();
            }
        }

        public class OpticsOrderingCursor : IRowCursor
        {
            readonly OpticsOrderingState _view;
            readonly IRowCursor _inputCursor;
            readonly int _colOrdering;
            readonly int _colReachability;
            readonly int _colCore;
            readonly int _colName;

            public OpticsOrderingCursor(OpticsOrderingState view, IRowCursor cursor)
            {
                _view = view;
                _colOrdering = view.Source.Schema.ColumnCount;
                _colReachability = _colOrdering + 1;
                _colCore = _colReachability + 1;
                _colName = _colCore + 1;
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
                if (col < _view.Source.Schema.ColumnCount)
                    return _inputCursor.GetGetter<TValue>(col);
                else if (col == _view.Source.Schema.ColumnCount) // Ordering
                    return GetGetterOrdering() as ValueGetter<TValue>;
                else if (col == _view.Source.Schema.ColumnCount + 1) // Reachability Distance
                    return GetGetterReachabilityDistance() as ValueGetter<TValue>;
                else if (col == _view.Source.Schema.ColumnCount + 2) // Core Distance
                    return GetGetterCoreDistance() as ValueGetter<TValue>;
                else
                    throw new IndexOutOfRangeException();
            }

            ValueGetter<long> GetGetterOrdering()
            {
                return (ref long orderingId) =>
                {
                    orderingId = _view.GetMappedIndex(_inputCursor.Position);
                };
            }

            ValueGetter<float> GetGetterReachabilityDistance()
            {
                return (ref float rDist) =>
                {
                    long rowIndex = _view.GetMappedIndex(_inputCursor.Position);
                    rDist = _view.OpticsOrderingResults[rowIndex].reachability;
                };
            }

            ValueGetter<float> GetGetterCoreDistance()
            {
                return (ref float cDist) =>
                {
                    long rowIndex = _view.GetMappedIndex(_inputCursor.Position);
                    cDist = _view.OpticsOrderingResults[rowIndex].core;
                };
            }
        }

        #endregion
    }
}
