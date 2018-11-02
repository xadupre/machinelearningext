
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
//using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model.Onnx;
//using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.Runtime.Tools;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Trainers.KMeans;
//using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Transforms;
using Scikit.ML.Clustering;
using Scikit.ML.FeaturesTransforms;
using Scikit.ML.ModelSelection;
using Scikit.ML.MultiClass;
using Scikit.ML.NearestNeighbors;
using Scikit.ML.OnnxHelper;
using Scikit.ML.PipelineGraphTraining;
using Scikit.ML.PipelineGraphTransforms;
using Scikit.ML.PipelineLambdaTransforms;
using Scikit.ML.PipelineTraining;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.ProductionPrediction;
using Scikit.ML.RandomTransforms;
using Scikit.ML.TimeSeries;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
namespace TestProfileBenchmark
{

    public static class DynamicCSFunctions_example_iris
    {


        public class IrisObservation
        {
            [Column("0")]
            [ColumnName("Label")]
            public string Label;

            [Column("1")]
            public float Sepal_length;

            [Column("2")]
            public float Sepal_width;

            [Column("3")]
            public float Petal_length;

            [Column("4")]
            public float Petal_width;
        }

        public class IrisPrediction
        {
            public uint PredictedLabel;

            [VectorType(4)]
            public float[] Score;
        }

        public static void example_iris()
        {
            var iris = "iris.txt";

            using (var env = new ConsoleEnvironment())
            {
                var reader = new TextLoader(env,
                                    new TextLoader.Arguments()
                                    {
                                        Separator = "	",
                                        HasHeader = true,
                                        Column = new[] {
                                    new TextLoader.Column("Label", DataKind.R4, 0),
                                    new TextLoader.Column("Sepal_length", DataKind.R4, 1),
                                    new TextLoader.Column("Sepal_width", DataKind.R4, 2),
                                    new TextLoader.Column("Petal_length", DataKind.R4, 3),
                                    new TextLoader.Column("Petal_width", DataKind.R4, 4),
                                        }
                                    });

                var pipeline = new ColumnConcatenatingEstimator(env, "Features", "Sepal_length", "Sepal_width", "Petal_length", "Petal_width")
                       .Append(new KMeansPlusPlusTrainer(env, "Features", clustersCount: 3));

                IDataView trainingDataView = reader.Read(new MultiFileSource(iris));
                var model = pipeline.Fit(trainingDataView);

                var obs = new IrisObservation()
                {
                    Sepal_length = 3.3f,
                    Sepal_width = 1.6f,
                    Petal_length = 0.2f,
                    Petal_width = 5.1f,
                };

                var engine = model.MakePredictionFunction<IrisObservation, IrisPrediction>(env);
                var res = engine.Predict(obs);
                Console.WriteLine("PredictedLabel: {0}", res.PredictedLabel);
                Console.WriteLine("Score: {0}", string.Join(", ", res.Score.Select(c => c.ToString())));
            }
        }

    }
}