// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;


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
        public void TestQ_KMmeansEntryPointAPI()
        {
            var iris = Scikit.ML.TestHelper.FileHelper.GetTestFile("iris.txt");

            var pipeline = new LearningPipeline();
            pipeline.Add(new Microsoft.ML.Data.TextLoader(iris).CreateFrom<IrisObservation>(separator: '\t'));
            pipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            pipeline.Add(new Microsoft.ML.Trainers.KMeansPlusPlusClusterer());
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

        public static ITrainer CreateTrainer(IHostEnvironment env, string settings, params object[] extraArgs)
        {
            var sc = SubComponent.Parse<ITrainer, SignatureTrainer>(settings);
            var inst = sc.CreateInstance(env, extraArgs);
            return inst;
        }

        [TestMethod]
        public void TestQ_KMmeasInnerAPI()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var iris = Scikit.ML.TestHelper.FileHelper.GetTestFile("iris.txt");
            var outModelFilePath = Scikit.ML.TestHelper.FileHelper.GetOutputFile("outModelFilePath.zip", methodName);

            var env = new TlcEnvironment(conc: 1);
            var data = env.CreateLoader("Text{col=Label:R4:0 col=Sepal_length:R4:1 col=Sepal_width:R4:2 col=Petal_length:R4:3 col=Petal_width:R4:4}",
                                        new MultiFileSource(iris));
            var conc = env.CreateTransform("Concat{col=Features:Sepal_length,Sepal_width}", data);
            var roleMap = env.CreateExamples(conc, "Features", "Label");
            var trainer = CreateTrainer(env, "km");
            IPredictor model;
            using (var ch = env.Start("Train"))
            {
                model = TrainUtils.Train(env, ch, roleMap, trainer, "KM", null, 0);
                ch.Done();
            }

            using (var ch = env.Start("Save"))
            {
                using (var fs = File.Create(outModelFilePath))
                    TrainUtils.SaveModel(env, ch, fs, model, roleMap);
                ch.Done();
            }

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

        [TestMethod]
        public void TestQ_KMeamsEntryPointAPIWithDataFrame()
        {
            var iris = Scikit.ML.TestHelper.FileHelper.GetTestFile("iris.txt");
            var df = Scikit.ML.DataManipulation.DataFrame.ReadCsv(iris, sep: '\t', dtypes: new Microsoft.ML.Runtime.Data.DataKind?[] { Microsoft.ML.Runtime.Data.DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new Scikit.ML.PipelineHelper.GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Microsoft.ML.Trainers.KMeansPlusPlusClusterer());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = Scikit.ML.DataManipulation.DataFrame.ReadView(predictions);
            Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 13));
        }
    }
}
