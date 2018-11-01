// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Text;
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

using TensorFlowTransform = Microsoft.ML.Transforms.TensorFlowTransform;


namespace Scikit.ML.ScikitAPI
{
    public static class ComponentHelper
    {
        /// <summary>
        /// Register one assembly.
        /// </summary>
        /// <param name="env">environment</param>
        /// <param name="a">assembly</param>
        public static void AddComponent(IHostEnvironment env, Assembly a)
        {
            try
            {
                env.ComponentCatalog.RegisterAssembly(a);
            }
            catch(Exception e)
            {
                throw new Exception($"Unable to register assembly '{a.FullName}' due to '{e}'.");
            }
        }

        /// <summary>
        /// Register standard assemblies from Microsoft.ML and Scikit.ML.
        /// </summary>
        /// <param name="env">environment</param>
        public static void AddStandardComponents(IHostEnvironment env)
        {
            AddComponent(env, typeof(TextLoader).Assembly);
            AddComponent(env, typeof(LinearPredictor).Assembly);
            AddComponent(env, typeof(CategoricalTransform).Assembly);
            AddComponent(env, typeof(FastTreeBinaryPredictor).Assembly);
            AddComponent(env, typeof(EnsemblePredictor).Assembly);
            AddComponent(env, typeof(KMeansPredictor).Assembly);
            AddComponent(env, typeof(LightGbm).Assembly);
            AddComponent(env, typeof(OlsLinearRegressionPredictor).Assembly);
            AddComponent(env, typeof(PcaPredictor).Assembly);
            AddComponent(env, typeof(SlidingWindowTransform).Assembly);
            AddComponent(env, typeof(TextFeaturizingEstimator).Assembly);
            AddComponent(env, typeof(TensorFlowTransform).Assembly);
            // ext
            AddComponent(env, typeof(DBScan).Assembly);
            AddComponent(env, typeof(DeTrendTransform).Assembly);
            AddComponent(env, typeof(PolynomialTransform).Assembly);
            AddComponent(env, typeof(PredictTransform).Assembly);
            AddComponent(env, typeof(NearestNeighborsBinaryClassificationTrainer).Assembly);
            AddComponent(env, typeof(MultiToBinaryPredictor).Assembly);
            AddComponent(env, typeof(TaggedPredictTransform).Assembly);
            AddComponent(env, typeof(AppendViewTransform).Assembly);
            AddComponent(env, typeof(PrePostProcessPredictor).Assembly);
            AddComponent(env, typeof(PassThroughTransform).Assembly);
            AddComponent(env, typeof(ResampleTransform).Assembly);
            AddComponent(env, typeof(SplitTrainTestTransform).Assembly);
            AddComponent(env, typeof(ValueMapperPredictionEngineFloat).Assembly);
        }
    }
}
