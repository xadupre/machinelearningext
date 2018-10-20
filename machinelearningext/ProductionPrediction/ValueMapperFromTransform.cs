// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Converts a <see cref="IDataTransform" /> into a <see cref="ValueMapper" />.
    /// This mapper is far from being efficient (execution time).
    /// The mapper creates a new view, a new iterator, serializes and deserializes the transform for each row.
    /// The serialization can be avoided by exposing a setter on _source.
    /// </summary>
    public class ValueMapperFromTransform<TRowInput, TRowOutput> : IValueMapper, IDisposable
        where TRowInput : class, IClassWithGetter<TRowInput>, new()
        where TRowOutput : class, IClassWithSetter<TRowOutput>, new()
    {
        readonly IDataTransform _transform;
        readonly IDataView _sourceToReplace;
        readonly IHostEnvironment _env;
        IHostEnvironment _computeEnv;
        readonly bool _getterEachTime;
        readonly bool _disposeEnv;

        public ColumnType InputType => null;
        public ColumnType OutputType => null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="transform">transform to convert</param>
        /// <param name="sourceToReplace">source to replace</param>
        /// <param name="getterEachTime">create the getter for each computation</param>
        /// <param name="conc">number of concurrency threads</param>
        public ValueMapperFromTransform(IHostEnvironment env, IDataTransform transform,
                                        IDataView sourceToReplace = null,
                                        bool getterEachTime = false, int conc = 1)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(transform);
            _env = env;
            _getterEachTime = getterEachTime;
            _transform = transform;
            _sourceToReplace = sourceToReplace;
            _disposeEnv = conc > 0;
            _computeEnv = _disposeEnv ? new PassThroughEnvironment(env, conc: conc, verbose: false) : env;
        }

        public void Dispose()
        {
            if (_disposeEnv)
            {
                (_computeEnv as PassThroughEnvironment).Dispose();
                _computeEnv = null;
            }
        }

        delegate void DelegateSetterRow<TDst, TValue>(ref TDst row, ref TValue value);

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            if (typeof(TSrc) != typeof(TRowInput))
                throw _env.Except($"Cannot create a mapper with input type {typeof(TSrc)} != {typeof(TRowInput)} (expected).");

            ValueMapper<TSrc, TDst> res;
            if (typeof(TDst) == typeof(TRowOutput))
                res = GetMapperRow() as ValueMapper<TSrc, TDst>;
            else
                res = GetMapperColumn<TDst>() as ValueMapper<TSrc, TDst>;
            if (res == null)
                throw _env.ExceptNotSupp($"Unable to create mapper from {typeof(TSrc)} to {typeof(TDst)}.");
            return res;
        }

        /// <summary>
        /// Returns a getter on particuler column.
        /// </summary>
        ValueMapper<TRowInput, TDst> GetMapperColumn<TDst>()
        {
            throw _env.ExceptNotImpl("Not implemented yet as it is missing the column index.");
        }

        ValueMapper<TRowInput, TRowOutput> GetMapperRow()
        {
            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(_transform);
            var schema = SchemaDefinition.Create(typeof(TRowOutput), SchemaDefinition.Direction.Read);

            if (_getterEachTime)
            {
                return (ref TRowInput src, ref TRowOutput dst) =>
                {
                    var inputView = new TemporaryViewCursorRow<TRowInput>(src, null, firstView.Schema, overwriteRowGetter: GetterSetterHelper.GetGetter<TRowInput>());
                    // We assume all columns are needed, otherwise they should be removed.
                    using (var inputCursor = inputView.GetRowCursor(i => true))
                    {
                        _env.AssertValue(inputCursor);

                        // This is extremely time consuming as the transform is serialized and deserialized.
                        var outputView = _sourceToReplace == _transform.Source
                                            ? ApplyTransformUtils.ApplyTransformToData(_env, _transform, inputView)
                                            : ApplyTransformUtils.ApplyAllTransformsToData(_env, _transform, inputView, _sourceToReplace);

                        using (var cur = outputView.GetRowCursor(i => true))
                        {
                            Delegate[] dels;
                            try
                            {
                                dels = new TRowOutput().GetCursorGetter(cur);
                            }
                            catch (InvalidOperationException e)
                            {
                                throw new InvalidOperationException($"Unable to create getter for the schema\n{SchemaHelper.ToString(cur.Schema)}", e);
                            }

                            cur.MoveNext();
                            dst.Set(dels);
                        }
                    }
                };
            }
            else
            {
                var inputView = new InfiniteLoopViewCursorRow<TRowInput>(null, firstView.Schema, overwriteRowGetter: GetterSetterHelper.GetGetter<TRowInput>());

                // This is extremely time consuming as the transform is serialized and deserialized.
                var outputView = _sourceToReplace == _transform.Source
                                    ? ApplyTransformUtils.ApplyTransformToData(_computeEnv, _transform, inputView)
                                    : ApplyTransformUtils.ApplyAllTransformsToData(_computeEnv, _transform, inputView, _sourceToReplace);

                // We assume all columns are needed, otherwise they should be removed.
                using (var cur = outputView.GetRowCursor(i => true))
                {
                    Delegate[] dels;
                    try
                    {
                        dels = new TRowOutput().GetCursorGetter(cur);
                    }
                    catch (InvalidOperationException e)
                    {
                        throw new InvalidOperationException($"Unable to create getter for the schema\n{SchemaHelper.ToString(cur.Schema)}", e);
                    }

                    return (ref TRowInput src, ref TRowOutput dst) =>
                    {
                        inputView.Set(ref src);
                        cur.MoveNext();
                        dst.Set(dels);
                    };
                }
            }
        }
    }
}
