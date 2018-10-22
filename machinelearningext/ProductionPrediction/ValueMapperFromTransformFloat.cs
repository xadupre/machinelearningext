// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
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
    public class ValueMapperFromTransformFloat<TColValue> : IValueMapper, IDisposable
    {
        public ColumnType InputType { get { return _transform.Source.Schema.GetColumnType(_inputIndex); } }
        public ColumnType OutputType { get { return _outputType; } }

        readonly IDataTransform _transform;
        readonly IDataView _sourceToReplace;
        readonly IHostEnvironment _env;
        readonly string _outputColumn;
        readonly int _inputIndex;
        readonly ColumnType _outputType;
        IHostEnvironment _computeEnv;
        readonly bool _getterEachTime;
        readonly bool _disposeEnv;
        readonly bool _ignoreOtherColumn;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">environment like ConsoleEnvironment</param>
        /// <param name="transform">transform to convert</param>
        /// <param name="inputColumn">input column of the mapper</param>
        /// <param name="outputColumn">output column of the mapper</param>
        /// <param name="sourceToReplace">source to replace</param>
        /// <param name="getterEachTime">create the getter for each computation</param>
        /// <param name="conc">number of concurrency threads</param>
        /// <param name="ignoreOtherColumn">ignore other columns instead of raising an exception if they are requested</param>
        public ValueMapperFromTransformFloat(IHostEnvironment env, IDataTransform transform,
                                             string inputColumn, string outputColumn,
                                             IDataView sourceToReplace = null,
                                             bool getterEachTime = false, int conc = 1,
                                             bool ignoreOtherColumn = false)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(transform);
            _env = env;
            _getterEachTime = getterEachTime;
            _transform = transform;
            _sourceToReplace = sourceToReplace;
            _outputColumn = outputColumn;
            _ignoreOtherColumn = ignoreOtherColumn;

            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(transform);

            int index;
            if (!firstView.Schema.TryGetColumnIndex(inputColumn, out index))
                throw env.Except("Unable to find column '{0}' in input schema '{1}'.",
                    inputColumn, SchemaHelper.ToString(firstView.Schema));
            _inputIndex = index;
            if (!transform.Schema.TryGetColumnIndex(outputColumn, out index))
                throw env.Except("Unable to find column '{0}' in output schema '{1}'.",
                    outputColumn, SchemaHelper.ToString(transform.Schema));
            _outputType = _transform.Schema.GetColumnType(index);

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

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            var firstView = _sourceToReplace ?? DataViewHelper.GetFirstView(_transform);
            if (_getterEachTime)
            {
                return (ref TSrc src, ref TDst dst) =>
                {
                    var inputView = new TemporaryViewCursorColumn<TSrc>(src, _inputIndex, firstView.Schema, ignoreOtherColumn: _ignoreOtherColumn);
                    using (var inputCursor = inputView.GetRowCursor(i => i == _inputIndex))
                    {
                        _env.AssertValue(inputCursor);

                        // This is extremely time consuming as the transform is serialized and deserialized.
                        var outputView = _sourceToReplace == _transform.Source
                                            ? ApplyTransformUtils.ApplyTransformToData(_env, _transform, inputView)
                                            : ApplyTransformUtils.ApplyAllTransformsToData(_env, _transform, inputView, _sourceToReplace);

                        int index;
                        if (!outputView.Schema.TryGetColumnIndex(_outputColumn, out index))
                            throw _env.Except("Unable to find column '{0}' in output schema.", _outputColumn);
                        int newOutputIndex = index;

                        using (var cur = outputView.GetRowCursor(i => i == newOutputIndex))
                        {
                            cur.MoveNext();
                            var getter = cur.GetGetter<TDst>(newOutputIndex);
                            getter(ref dst);
                        }
                    }
                };
            }
            else
            {
                var inputView = new InfiniteLoopViewCursorColumn<TSrc>(_inputIndex, firstView.Schema, ignoreOtherColumn: _ignoreOtherColumn);

                // This is extremely time consuming as the transform is serialized and deserialized.
                var outputView = _sourceToReplace == _transform.Source
                                    ? ApplyTransformUtils.ApplyTransformToData(_computeEnv, _transform, inputView)
                                    : ApplyTransformUtils.ApplyAllTransformsToData(_computeEnv, _transform, inputView, _sourceToReplace);

                int index;
                if (!outputView.Schema.TryGetColumnIndex(_outputColumn, out index))
                    throw _env.Except("Unable to find column '{0}' in output schema.", _outputColumn);
                int newOutputIndex = index;

                using (var cur = outputView.GetRowCursor(i => i == newOutputIndex))
                {
                    var getter = cur.GetGetter<TDst>(newOutputIndex);
                    if (getter == null)
                        throw _env.Except("Unable to get a getter on the transform for type {0}", default(TDst).GetType());
                    return (ref TSrc src, ref TDst dst) =>
                    {
                        inputView.Set(ref src);
                        cur.MoveNext();
                        getter(ref dst);
                    };
                }
            }
        }
    }
}
