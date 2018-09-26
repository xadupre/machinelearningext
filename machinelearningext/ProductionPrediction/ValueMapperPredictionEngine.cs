// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.ProductionPrediction
{
    public class FloatVectorInput
    {
        [VectorType(1)]
        public float[] Features;
    }

    /// <summary>
    /// Creates a prediction engine which does not create getters each time.
    /// It is much faster as it does not recreate getter for every observation.
    /// </summary>
    /// <typeparam name="TRowInput">a row</typeparam>
    public class ValueMapperPredictionEngine
    {

        readonly IHostEnvironment _env;
        readonly IDataView _transforms;
        readonly Predictor _predictor;
        readonly ValueMapper<VBuffer<float>, float> _mapper;
        readonly ValueMapper<VBuffer<float>, VBuffer<float>> _mapperVector;

        public ValueMapperPredictionEngine()
        {
            throw Contracts.Except("Use arguments.");
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelName">filename</param>
        /// <param name="output">name of the output column</param>
        /// <param name="getterEachTime">true to create getter each time a prediction is made (multithrading is allowed) or not (no multithreading)</param>
        /// <param name="outputIsFloat">output is a gloat (true) or a vector of floats (false)</param>
        /// <param name="conc">number of concurrency threads</param>
        public ValueMapperPredictionEngine(IHostEnvironment env, string modelName,
                string output = "Probability", bool getterEachTime = false,
                bool outputIsFloat = true, int conc = 1) :
            this(env, File.OpenRead(modelName), output, getterEachTime, outputIsFloat, conc)
        {
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="modelStream">stream</param>
        /// <param name="output">name of the output column</param>
        /// <param name="getterEachTime">true to create getter each time a prediction is made (multithrading is allowed) or not (no multithreading)</param>
        /// <param name="outputIsFloat">output is a gloat (true) or a vector of floats (false)</param>
        /// <param name="conc">number of concurrency threads</param>
        public ValueMapperPredictionEngine(IHostEnvironment env, Stream modelStream,
                string output = "Probability", bool getterEachTime = false,
                bool outputIsFloat = true, int conc = 1)
        {
            _env = env;
            if (_env == null)
                throw Contracts.Except("env must not be null");
            var inputs = new FloatVectorInput[0];
            var view = ComponentCreation.CreateStreamingDataView<FloatVectorInput>(_env, inputs);

            long modelPosition = modelStream.Position;
            _predictor = ComponentCreation.LoadPredictorOrNull(_env, modelStream);
            if (_predictor == null)
                throw _env.Except("Unable to load a model.");
            modelStream.Seek(modelPosition, SeekOrigin.Begin);
            _transforms = ComponentCreation.LoadTransforms(_env, modelStream, view);
            if (_transforms == null)
                throw _env.Except("Unable to load a model.");

            var features = "Features";
            var data = _env.CreateExamples(_transforms, features);
            if (data == null)
                throw _env.Except("Cannot create rows.");
            var scorer = _env.CreateDefaultScorer(data, _predictor);
            if (scorer == null)
                throw _env.Except("Cannot create a scorer.");

            var valueMapper = new ValueMapperFromTransform<VBuffer<float>>(_env,
                                scorer, view, features, output, null, getterEachTime, conc);
            if (valueMapper == null)
                throw _env.Except("Cannot create a mapper.");
            if (outputIsFloat)
            {
                _mapper = valueMapper.GetMapper<VBuffer<float>, float>();
                _mapperVector = null;
            }
            else
            {
                _mapper = null;
                _mapperVector = valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();
            }
        }

        /// <summary>
        /// Produces prediction assuming the input accepts a features vectors as inputs.
        /// </summary>
        /// <param name="features"></param>
        /// <returns></returns>
        public float Predict(float[] features)
        {
            if (_mapper == null)
                throw _env.Except("The mapper is outputting a vector not a float.");
            float res = 0f;
            var buf = new VBuffer<float>(features.Length, features);
            _mapper(ref buf, ref res);
            return res;
        }

        /// <summary>
        /// Produces prediction assuming the input accepts a features vectors as inputs.
        /// </summary>
        /// <param name="features"></param>
        /// <returns></returns>
        public float[] PredictVector(float[] features)
        {
            if (_mapperVector == null)
                throw _env.Except("The mapper is outputting a float not a vector.");
            VBuffer<float> res = new VBuffer<float>();
            var buf = new VBuffer<float>(features.Length, features);
            _mapperVector(ref buf, ref res);
            if (!res.IsDense)
                throw _env.Except("The output of the predictor or transform must be dense.");
            return res.DenseValues().ToArray();
        }
    }
}
