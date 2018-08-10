// See the LICENSE file in the project root for more information.

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
    /// The creation of a new view and a new iterator can be avoided if the function GetRowCursor
    /// ensures it returns a iterator in the same thread (no parallelization).
    /// 
    /// To get that, the parameter singleThread must be set to true.
    /// </summary>
    public class ValueMapperFromTransform<TColValue> : IValueMapper
    {
        public ColumnType InputType { get { return _transform.Source.Schema.GetColumnType(_inputIndex); } }
        public ColumnType OutputType { get { return _outputType; } }

        readonly IDataTransform _transform;
        readonly IDataView _source;
        readonly int _nbSteps;
        readonly IHostEnvironment _env;
        readonly string _outputColumn;
        readonly int _inputIndex;
        readonly ColumnType _outputType;
        readonly IRowCursor _currentInputCursor;
        readonly TlcEnvironment _singleThreadEnv;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="env">TLC</param>
        /// <param name="transform">transform to convert</param>
        /// <param name="source">source to replace</param>
        /// <param name="inputColumn">input column of the mapper</param>
        /// <param name="outputColumn">output column of the mapper</param>
        /// <param name="currentCursor">if you need access to other columns than the input one, an addition cursor must be given</param>
        /// <param name="singleThread">See documentation of the class.</param>
        public ValueMapperFromTransform(IHostEnvironment env, IDataTransform transform, IDataView source,
                                        string inputColumn, string outputColumn, IRowCursor currentCursor,
                                        bool singleThread = false)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(transform);
            _env = env;
            _transform = transform;
            if (source == null)
            {
                _source = transform.Source;
                _nbSteps = 1;
            }
            else
            {
                _nbSteps = 2;
                _source = source;
            }
            _outputColumn = outputColumn;
            _currentInputCursor = currentCursor;

            int index;
            if (!_source.Schema.TryGetColumnIndex(inputColumn, out index))
                throw env.Except("Unable to find column '{0}' in input schema '{1}'.",
                    inputColumn, SchemaHelper.ToString(_source.Schema));
            _inputIndex = index;
            if (!transform.Schema.TryGetColumnIndex(outputColumn, out index))
                throw env.Except("Unable to find column '{0}' in output schema '{1}'.",
                    outputColumn, SchemaHelper.ToString(transform.Schema));
            _outputType = _transform.Schema.GetColumnType(index);

            _singleThreadEnv = singleThread ? new TlcEnvironment(conc: 1, verbose: false) : null;
        }

        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            if (_singleThreadEnv == null)
            {
                return (ref TSrc src, ref TDst dst) =>
                {
                    var inputView = new TemporaryViewCursor<TSrc>(src, _inputIndex, _currentInputCursor, _source.Schema);
                    // Let's keep it simple: we ask for all columns.
                    using (var inputCursor = inputView.GetRowCursor(i => i == _inputIndex))
                    {
                        _env.AssertValue(inputCursor);

                        // This is extremely time consuming as the transform is serialized and deserialized.
                        var outputView = _nbSteps == 1
                                            ? ApplyTransformUtils.ApplyTransformToData(_env, _transform, inputView)
                                            : ApplyTransformUtils.ApplyAllTransformsToData(_env, _transform, inputView, _source);

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
                var inputView = new InfiniteLoopViewCursor<TSrc>(_inputIndex, _currentInputCursor, _source.Schema);

                // This is extremely time consuming as the transform is serialized and deserialized.
                var outputView = _nbSteps == 1
                                    ? ApplyTransformUtils.ApplyTransformToData(_singleThreadEnv, _transform, inputView)
                                    : ApplyTransformUtils.ApplyAllTransformsToData(_singleThreadEnv, _transform, inputView, _source);

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

