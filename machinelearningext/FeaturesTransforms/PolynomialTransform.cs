// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Scikit.ML.PipelineHelper;

using LoadableClassAttribute = Microsoft.ML.Runtime.LoadableClassAttribute;
using SignatureDataTransform = Microsoft.ML.Runtime.Data.SignatureDataTransform;
using SignatureLoadDataTransform = Microsoft.ML.Runtime.Data.SignatureLoadDataTransform;
using PolynomialTransform = Scikit.ML.FeaturesTransforms.PolynomialTransform;

[assembly: LoadableClass(PolynomialTransform.Summary, typeof(PolynomialTransform),
    typeof(PolynomialTransform.Arguments), typeof(SignatureDataTransform),
    PolynomialTransform.LongName, PolynomialTransform.LoaderSignature, PolynomialTransform.ShortName)]

[assembly: LoadableClass(PolynomialTransform.Summary, typeof(PolynomialTransform),
    null, typeof(SignatureLoadDataTransform),
    PolynomialTransform.LongName, PolynomialTransform.LoaderSignature, PolynomialTransform.ShortName)]


namespace Scikit.ML.FeaturesTransforms
{
    /// <summary>
    /// Multiplies features, build polynomial features x1, x1^2, x1x2, x2, x2^2...
    /// </summary>
    public class PolynomialTransform : IDataTransform
    {
        #region identification

        public const string LoaderSignature = "PolynomialTransform";  // Not more than 24 letters.
        public const string Summary = "Multiplies features, build polynomial features x1, x1^2, x1x2, x2, x2^2... " +
                                      "The output should be cached otherwise the transform will recompute the features " +
                                      "each time it is needed. Use CacheTransform.";
        public const string RegistrationName = LoaderSignature;
        public const string ShortName = "Poly";
        public const string LongName = "Polynomial Transform";

        /// <summary>
        /// Identify the object for dynamic instantiation.
        /// This is also used to track versionning when serializing and deserializing.
        /// </summary>
        static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "POLYTRAN",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        #endregion

        #region dimensions

        static Func<int, int>[] Total = new Func<int, int>[] {
                                k => 0,
                                k => k,
                                k => k * (k + 1) / 2,
                                k => k * (k * k + 3 * k + 2) / 6
                };
        static Func<int, int>[] TotalCumulated = new Func<int, int>[] {
                                k => 0,
                                k => k,
                                k => Total[1](k) + Total[2](k),
                                k => Total[1](k) + Total[2](k) + Total[3](k)
                };

        #endregion

        #region parameters / command line

        /// <summary>
        /// Parameters which defines the transform.
        /// </summary>
        public class Arguments
        {
            [Argument(ArgumentType.MultipleUnique, HelpText = "Features columns (a vector)", ShortName = "col")]
            public Column1x1[] columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Highest degree of the polynomial features", ShortName = "d")]
            public int degree = 2;

            // This parameter is not used right now. We could imagine that the transform walk the through the data first
            // and determines and filters out polynomial features. That would add a training step.
            // In that case, transforms usual accepts a parameter which specifies the number of threads this training
            // steps could have.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of threads used to estimate allowed by the transform.", ShortName = "nt")]
            public int? numThreads;

            public void Write(ModelSaveContext ctx, IHost host)
            {
                ctx.Writer.Write(Column1x1.ArrayToLine(columns));
                ctx.Writer.Write(degree);
                ctx.Writer.Write(numThreads ?? -1);
            }

            public void Read(ModelLoadContext ctx, IHost host)
            {
                string sr = ctx.Reader.ReadString();
                columns = Column1x1.ParseMulti(sr);
                degree = ctx.Reader.ReadInt32();
                int nb = ctx.Reader.ReadInt32();
                numThreads = nb > 0 ? (int?)nb : null;
            }
        }

        #endregion

        #region internal members / accessors

        IDataView _input;
        IDataTransform _transform;          // templated transform (not the serialized version)
        Arguments _args;
        IHost _host;

        public IDataView Source => _input;
        public IHost Host => _host;

        #endregion

        #region public constructor / serialization / load / save

        public PolynomialTransform(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            _host = env.Register("PolynomialTransform");
            _host.CheckValue(args, "args");                 // Checks values are valid.
            _host.CheckValue(input, "input");
            _host.CheckValue(args.columns, "columns");
            _host.Check(args.degree > 1, "degree must be > 1");

            _input = input;

            int ind;
            foreach (var col in args.columns)
                if (!input.Schema.TryGetColumnIndex(col.Source, out ind))
                    throw _host.ExceptParam("columns", "Column '{0}' not found in schema.", col.Source);
            _args = args;
            _transform = CreateTemplatedTransform();
        }

        /// <summary>
        /// Static function to append the transform to an existing pipeline.
        /// Do not forget this otherwise the pipeline cannot be instantiated.
        /// </summary>
        public static PolynomialTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, "env");
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, "ctx");
            h.CheckValue(input, "input");
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new PolynomialTransform(h, ctx, input));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, "ctx");
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            _args.Write(ctx, _host);
        }

        private PolynomialTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(host, "host");
            Contracts.CheckValue(input, "input");
            _host = host;
            _input = input;
            _host.CheckValue(input, "input");
            _host.CheckValue(ctx, "ctx");
            _args = new Arguments();
            _args.Read(ctx, _host);
            _transform = CreateTemplatedTransform();
        }

        #endregion

        #region IDataTransform API

        public ISchema Schema { get { return _transform.Schema; } }
        public bool CanShuffle { get { return _input.CanShuffle; } }

        /// <summary>
        /// Same as the input data view.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            _host.AssertValue(Source, "_input");
            return Source.GetRowCount(lazy); // We do not add or remove any row. Same number of rows as the input.
        }

        /// <summary>
        /// If the function returns null or true, the method GetRowCursorSet
        /// needs to be implemented.
        /// </summary>
        protected bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            return true;
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            // Fun part we'll see later.
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            _host.AssertValue(_transform, "_transform");
            return _transform.GetRowCursorSet(out consolidator, predicate, n, rand);
        }

        #endregion

        #region transform own logic

        /// <summary>
        /// Create the internal transform (not serialized in the zip file).
        /// </summary>
        private IDataTransform CreateTemplatedTransform()
        {
            IDataTransform transform = null;

            // The column is a vector.
            int index = -1;
            foreach (var col in _args.columns)
                if (!_input.Schema.TryGetColumnIndex(col.Source, out index))
                    throw _host.Except("Unable to find '{0}'", col.Source);
            if (_args.columns.Length != 1)
                throw _host.Except("Only one column allowed not '{0}'.", _args.columns.Length);

            var typeCol = _input.Schema.GetColumnType(index);
            if (!typeCol.IsVector)
                throw _host.Except("Expected a vector as input.");
            typeCol = typeCol.AsVector.ItemType;

            // We may consider multiple types here, vector of float, uint, int...
            // Let's do float and int.
            switch (typeCol.RawKind)
            {
                case DataKind.R4:
                    transform = new PolynomialState<float>(_host, transform ?? Source, _args, (a, b) => a * b);
                    break;
                case DataKind.U4:
                    transform = new PolynomialState<UInt32>(_host, transform ?? Source, _args, (a, b) => a * b);
                    break;
                default:
                    throw Contracts.ExceptNotSupp("Type '{0}' is not handled yet.", typeCol.RawKind);
            }
            return transform;
        }

        #endregion

        #region State

        /// <summary>
        /// Templated transform which sorts rows based on one column.
        /// Some precisions: the serialized pipeline (in the zip) creates a memory pipeline (which process the data).
        /// Above is the serialized pipeline. Below the memory pipeline.
        /// TInput is a simple type (no vector).
        /// </summary>
        public class PolynomialState<TInput> : IDataTransform
        {
            IHost _host;
            IDataView _input;

            readonly ISchema _schema;
            readonly Arguments _args;
            readonly int _inputCol;
            readonly Func<TInput, TInput, TInput> _multiplication;

            // Unused fo the time begin. This might be required if the transform has a training steps.
            // We want this step to be executed only once when the next transform in the pipeline 
            // requires the output of this one.
            // object _lock;

            public IDataView Source => _input;
            public ISchema Schema => _schema;
            public IHost Host => _host;

            public PolynomialState(IHostEnvironment host, IDataView input, Arguments args, Func<TInput, TInput, TInput> multiplication)
            {
                _host = host.Register("PolynomialState");
                _host.CheckValue(input, "input");
                _input = input;
                // _lock = new object();
                _args = args;
                _multiplication = multiplication;
                var column = _args.columns[0];
                using (var ch = _host.Start("PolynomialState"))
                {
                    if (!input.Schema.TryGetColumnIndex(column.Source, out _inputCol))
                        throw _host.ExceptParam("inputColumn", "Column '{0}' not found in schema.", column.Source);
                    var type = input.Schema.GetColumnType(_inputCol);
                    if (!type.IsVector)
                        throw _host.Except("Input column type must be a vector.");
                    int dim = type.AsVector.DimCount;
                    if (dim > 1)
                        throw _host.Except("Input column type must be a vector of one dimension.");
                    int size = dim > 0 ? type.AsVector.GetDim(0) : 0;
                    if (size > 0)
                        size = TotalCumulated[_args.degree](size);
                    ch.Trace("PolynomialTransform {0}->{1}.", dim, size);

                    // We extend the input schema. The new type has the same type as the input.
                    _schema = new ExtendedSchema(input.Schema, new[] { column.Name }, new[] { new VectorType(type.AsVector.ItemType, size) });
                    ch.Done();
                }
            }

            public void Save(ModelSaveContext ctx)
            {
                // Needed by the API but does nothing.
            }

            public bool CanShuffle { get { return true; } }

            public long? GetRowCount(bool lazy = true)
            {
                return _input.GetRowCount(lazy);
            }

            /// <summary>
            /// When the last column is requested, we also need the column used to compute it.
            /// This function ensures that this column is requested when the last one is.
            /// </summary>
            bool PredicatePropagation(int col, Func<int, bool> predicate)
            {
                if (predicate(col))
                    return true;
                if (col == _inputCol)
                    return predicate(Source.Schema.ColumnCount);
                return predicate(col);
            }

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                if (predicate(_input.Schema.ColumnCount))
                {
                    var cursor = _input.GetRowCursor(i => PredicatePropagation(i, predicate), rand);
                    return new PolynomialCursor<TInput>(this, cursor, i => PredicatePropagation(i, predicate), _args, _inputCol, _multiplication);
                }
                else
                    // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                    return new SameCursor(_input.GetRowCursor(predicate, rand), Schema);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
            {
                if (predicate(_input.Schema.ColumnCount))
                {
                    var cursors = _input.GetRowCursorSet(out consolidator, i => PredicatePropagation(i, predicate), n, rand);
                    return cursors.Select(c => new PolynomialCursor<TInput>(this, c, predicate, _args, _inputCol, _multiplication)).ToArray();
                }
                else
                    // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                    return _input.GetRowCursorSet(out consolidator, predicate, n, rand)
                                 .Select(c => new SameCursor(c, Schema))
                                 .ToArray();
            }
        }

        #endregion

        #region Cursor

        class PolynomialCursor<TInput> : IRowCursor
        {
            readonly PolynomialState<TInput> _view;
            readonly IRowCursor _inputCursor;
            readonly Arguments _args;
            readonly Func<TInput, TInput, TInput> _multiplication;

            ValueGetter<VBuffer<TInput>> _inputGetter;

            public PolynomialCursor(PolynomialState<TInput> view, IRowCursor cursor, Func<int, bool> predicate,
                                    Arguments args, int column, Func<TInput, TInput, TInput> multiplication)
            {
                if (!predicate(column))
                    throw view.Host.ExceptValue("Required column is not generated by previous layers.");
                _view = view;
                _args = args;
                _inputCursor = cursor;
                _inputGetter = cursor.GetGetter<VBuffer<TInput>>(column);
                _multiplication = multiplication;
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

            public CursorState State => _inputCursor.State; // No change.
            public long Batch => _inputCursor.Batch;        // No change.
            public long Position => _inputCursor.Position;  // No change.
            public ISchema Schema => _view.Schema;          // No change.

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
                    return PolynomialBuilder() as ValueGetter<TValue>;
                // Otherwise, it is an error.
                else
                    throw Contracts.Except("Unexpected columns {0} > {1}.", col, _inputCursor.Schema.ColumnCount);
            }

            /// <summary>
            /// We compute the polynomial features.
            /// </summary>
            private ValueGetter<VBuffer<TInput>> PolynomialBuilder()
            {
                // VBuffer<TInput> is the internal representation of a vector.
                // It can be dense (TInput[]) or sparse.
                // If there are n features, we can expect sum(i=1, d) n^i / i! polynomial features.
                VBuffer<TInput> features = new VBuffer<TInput>();
                int degree = _args.degree;
                var values = new List<TInput>();
                var indices = new List<int>();
                int[] tempIndices = new int[3];

                Func<IEnumerable<int>, int, int> computeIndex = (IEnumerable<int> sparseIndices, int nbFeatures) =>
                {
                    int nb = 0;
                    foreach (var i in sparseIndices)
                    {
                        tempIndices[nb] = i;
                        nb += 1;
                    }
                    switch (nb)
                    {
                        case 1:
                            return tempIndices[0];
                        case 2:
                            int d1 = Total[1](nbFeatures);
                            int d2 = Total[2](nbFeatures);
                            return d1 + (d2 - Total[2](nbFeatures - tempIndices[0])) + (tempIndices[1] - tempIndices[0]);
                        case 3:
                            int d1_ = Total[1](nbFeatures);
                            int d2_ = Total[2](nbFeatures);
                            int d3_ = Total[3](nbFeatures);
                            int d1d = Total[2](nbFeatures - tempIndices[0]);
                            int d2d = Total[2](nbFeatures - tempIndices[1]);
                            return d1_ + d2_ +
                                   d1d - d2d + tempIndices[2] - tempIndices[1] + // part with N^2
                                   d3_ - Total[3](nbFeatures - tempIndices[0]);  // part with N^3

                        default:
                            throw Contracts.ExceptNotSupp("Level should be in [1, 3].");
                    }
                };

                return (ref VBuffer<TInput> polyfeat) =>
                {
                    _inputGetter(ref features);
                    int total;

                    if (features.IsDense)
                    {
                        var poly = EnumeratePosition(features.Count, degree)
                                            .Select(pos => pos.Select(p => features.Values[p]).Aggregate((a, b) => _multiplication(a, b)))
                                            .ToArray();
                        polyfeat = new VBuffer<TInput>(poly.Length, poly);
                    }
                    else
                    {
                        values.Clear();
                        indices.Clear();

                        foreach (var pos in EnumeratePosition(features.Count, degree))
                        {
                            values.Add(pos.Select(p => features.Values[p]).Aggregate((a, b) => _multiplication(a, b)));
                            indices.Add(computeIndex(pos.Select(p => features.Indices[p]), features.Length));
#if (DEBUG)
                            if (indices.Count > 1)
                            {
                                if (indices[indices.Count - 1] <= indices[indices.Count - 2])
                                    throw Contracts.Except("Inconsistency");
                            }
#endif
                        }
                        total = TotalCumulated[_args.degree](features.Length);
                        polyfeat = new VBuffer<TInput>(total, values.Count, values.ToArray(), indices.ToArray());
                    }
                };
            }

            static IEnumerable<int[]> EnumeratePosition(int nbfeatures, int degree)
            {
                if (degree >= 1)
                {
                    var position = new int[1];
                    for (int i = 0; i < nbfeatures; ++i)
                    {
                        position[0] = i;
                        yield return position;
                    }
                }
                if (degree >= 2)
                {
                    var position = new int[2];
                    for (int i = 0; i < nbfeatures; ++i)
                    {
                        position[0] = i;
                        for (int j = i; j < nbfeatures; ++j)
                        {
                            position[1] = j;
                            yield return position;
                        }
                    }
                }
                if (degree >= 3)
                {
                    var position = new int[3];
                    for (int i = 0; i < nbfeatures; ++i)
                    {
                        position[0] = i;
                        for (int j = i; j < nbfeatures; ++j)
                        {
                            position[1] = j;
                            for (int k = j; k < nbfeatures; ++k)
                            {
                                position[2] = k;
                                yield return position;
                            }
                        }
                    }
                }
                if (degree >= 4)
                    throw Contracts.ExceptNotImpl("Not implemented for a degree >= 4");
            }
        }

        #endregion
    }
}
