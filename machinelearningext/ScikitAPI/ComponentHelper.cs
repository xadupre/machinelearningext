// See the LICENSE file in the project root for more information.

using System;
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
            env.ComponentCatalog.RegisterAssembly(typeof(ValueMapperPredictionEngineFloat).Assembly);
        }
    }
}
