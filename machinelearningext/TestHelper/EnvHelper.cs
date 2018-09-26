// See the LICENSE file in the project root for more information.

using System.IO;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PCA;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.KMeans;
using Microsoft.ML.Runtime.Ensemble;
using Scikit.ML.Clustering;
using Scikit.ML.TimeSeries;
using Scikit.ML.FeaturesTransforms;
using Scikit.ML.PipelineLambdaTransforms;
using Scikit.ML.ModelSelection;
using Scikit.ML.MultiClass;
using Scikit.ML.NearestNeighbors;
using Scikit.ML.PipelineGraphTraining;
using Scikit.ML.PipelineGraphTransforms;
using Scikit.ML.PipelineTraining;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.RandomTransforms;


namespace Scikit.ML.TestHelper
{
    public static class EnvHelper
    {
        public static void AddStandardComponents(IHostEnvironment env)
        {
            env.ComponentCatalog.RegisterAssembly(typeof(TextLoader).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(LinearPredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(CategoricalTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(FastTreeBinaryPredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(EnsemblePredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(KMeansPredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(PcaPredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(TextTransform).Assembly);
            // ext
            env.ComponentCatalog.RegisterAssembly(typeof(DBScan).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(DeTrendTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(PolynomialTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(PredictTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(NearestNeighborsBinaryClassificationTrainer).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(MultiToBinaryPredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(TaggedPredictTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(AppendViewTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(PrePostProcessPredictor).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(PassThroughTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(ResampleTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(SplitTrainTestTransform).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(ValueMapperPredictionEngine).Assembly);
        }

        /// <summary>
        /// Creates a new environment. It should be done
        /// with <tt>using</tt>.
        /// </summary>
        public static ConsoleEnvironment NewTestEnvironment(int? seed = null, bool verbose = false,
                            MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                            int conc = 0, TextWriter outWriter = null, TextWriter errWriter = null)
        {
            if (!seed.HasValue)
                seed = 42;
            if (outWriter == null)
                outWriter = new StreamWriter(new MemoryStream());
            if (errWriter == null)
                errWriter = new StreamWriter(new MemoryStream());

            var env = new ConsoleEnvironment(seed, verbose, sensitivity, conc, outWriter, errWriter);
            AddStandardComponents(env);
            return env;
        }

        public static ConsoleEnvironment NewTestEnvironment(out StringWriter sout, out StringWriter serr,
                                                        int? seed = null, bool verbose = false,
                                                        MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                                                        int conc = 0)
        {
            var sb = new StringBuilder();
            sout = new StringWriter(sb);
            sb = new StringBuilder();
            serr = new StringWriter(sb);
            return NewTestEnvironment(seed: seed, verbose: verbose, sensitivity: sensitivity,
                                      conc: conc, outWriter: sout, errWriter: serr);
        }
    }
}
