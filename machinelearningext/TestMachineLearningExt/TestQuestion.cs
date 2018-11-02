// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.TestHelper;
using Scikit.ML.ProductionPrediction;
using Legacy = Microsoft.ML.Legacy;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestQuestion
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

        [TestMethod]
        public void TestEP_Q_KMeansEntryPointAPI_04()
        {
            var iris = FileHelper.GetTestFile("iris.txt");

            var pipeline = new Legacy.LearningPipeline();
            pipeline.Add(new Legacy.Data.TextLoader(iris).CreateFrom<IrisObservation>(separator: '\t', useHeader: true));
            pipeline.Add(new Legacy.Transforms.ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            pipeline.Add(new Legacy.Trainers.KMeansPlusPlusClusterer());
            var model = pipeline.Train<IrisObservation, IrisPrediction>();
            var obs = new IrisObservation()
            {
                Sepal_length = 3.3f,
                Sepal_width = 1.6f,
                Petal_length = 0.2f,
                Petal_width = 5.1f,
            };
            var predictions = model.Predict(obs);
            Assert.IsTrue(predictions.PredictedLabel != 0);
        }

        [TestMethod]
        public void TestEP_Q_KMeansEntryPointAPI_06()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            using (var env = new ConsoleEnvironment())
            {
                var reader = new TextLoader(env,
                                    new TextLoader.Arguments()
                                    {
                                        Separator = "\t",
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

                var prediction = model.MakePredictionFunction<IrisObservation, IrisPrediction>(env).Predict(obs);
                Assert.IsTrue(prediction.PredictedLabel != 0);

                var df = Scikit.ML.DataManipulation.DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });
                var prediction2 = model.MakePredictionFunctionDataFrame(env, df.Schema);
                var df2 = Scikit.ML.DataManipulation.DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });
                var df3 = prediction2.Predict(df2);
                Assert.AreEqual(df.Shape[0], df3.Shape[0]);
            }
        }

        public static ITrainer CreateTrainer(IHostEnvironment env, string settings, params object[] extraArgs)
        {
            var sc = ScikitSubComponent.Parse<ITrainer, SignatureTrainer>(settings);
            var inst = sc.CreateInstance(env, extraArgs);
            return inst;
        }

        [TestMethod]
        public void TestI_Q_KMeansInnerAPI()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var iris = FileHelper.GetTestFile("iris.txt");
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                ComponentHelper.AddStandardComponents(env);

                var data = env.CreateLoader("Text{col=Label:R4:0 col=Sepal_length:R4:1 col=Sepal_width:R4:2 " +
                                            "col=Petal_length:R4:3 col=Petal_width:R4:4 header=+}",
                                            new MultiFileSource(iris));
                var conc = env.CreateTransform("Concat{col=Features:Sepal_length,Sepal_width}", data);
                var roleMap = env.CreateExamples(conc, "Features", "Label");
                var trainer = CreateTrainer(env, "km");
                IPredictor model;
                using (var ch = env.Start("Train"))
                    model = TrainUtils.Train(env, ch, roleMap, trainer, null, 0);

                using (var ch = env.Start("Save"))
                using (var fs = File.Create(outModelFilePath))
                    TrainUtils.SaveModel(env, ch, fs, model, roleMap);

                var obs = new IrisObservation()
                {
                    Sepal_length = 3.3f,
                    Sepal_width = 1.6f,
                    Petal_length = 0.2f,
                    Petal_width = 5.1f,
                };

                using (var fs = File.OpenRead(outModelFilePath))
                {
                    var engine = env.CreatePredictionEngine<IrisObservation, IrisPrediction>(fs);
                    var predictions = engine.Predict(obs);
                    Assert.IsTrue(predictions.PredictedLabel != 0);
                }
            }
        }

        [TestMethod]
        public void TestEP_Q_KMeansEntryPointAPIWithDataFrame()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = Scikit.ML.DataManipulation.DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new Legacy.Transforms.ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Legacy.Trainers.KMeansPlusPlusClusterer());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = Scikit.ML.DataManipulation.DataFrameIO.ReadView(predictions);
            Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 13));
        }

        [TestMethod]
        public void TestI_Q_KMeansInnerAPIWithDataFrame()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var iris = FileHelper.GetTestFile("iris.txt");
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                ComponentHelper.AddStandardComponents(env);

                var df = Scikit.ML.DataManipulation.DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });
                var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
                var roleMap = env.CreateExamples(conc, "Feature", label: "Label");
                var trainer = CreateTrainer(env, "km");
                IPredictor model;
                using (var ch = env.Start("test"))
                    model = TrainUtils.Train(env, ch, roleMap, trainer, null, 0);

                using (var ch = env.Start("Save"))
                using (var fs = File.Create(outModelFilePath))
                    TrainUtils.SaveModel(env, ch, fs, model, roleMap);

                Predictor pred;
                using (var fs = File.OpenRead(outModelFilePath))
                    pred = env.LoadPredictorOrNull(fs);

#pragma warning disable CS0618
                var scorer = ScoreUtils.GetScorer(pred.GetPredictorObject() as IPredictor, roleMap, env, null);
#pragma warning restore CS0618
                var dfout = Scikit.ML.DataManipulation.DataFrameIO.ReadView(scorer);
                Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 13));
            }
        }
    }
}
