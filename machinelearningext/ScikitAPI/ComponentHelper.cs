// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Tools;
using Scikit.ML.DataManipulation;
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
using Scikit.ML.ScikitAPI;


namespace Scikit.ML.ScikitAPI
{
    public static class ComponentHelper
    {
        private static void AddComponent(IHostEnvironment env, Assembly a)
        {
            try
            {
                env.ComponentCatalog.RegisterAssembly(a);
            }
            catch(Exception e)
            {
                throw new Exception($"Unable to register assembly {a.FullName} due to ${e}.");
            }
        }

        public static void AddStandardComponents(IHostEnvironment env)
        {
            AddComponent(env, typeof(TextLoader).Assembly);
            AddComponent(env, typeof(LinearPredictor).Assembly);
            AddComponent(env, typeof(CategoricalTransform).Assembly);
            AddComponent(env, typeof(FastTreeBinaryPredictor).Assembly);
            AddComponent(env, typeof(EnsemblePredictor).Assembly);
            AddComponent(env, typeof(KMeansPredictor).Assembly);
            AddComponent(env, typeof(PcaPredictor).Assembly);
            AddComponent(env, typeof(TextTransform).Assembly);
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
