﻿// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Microsoft.ML;
using Scikit.ML.TestHelper;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Transforms;
using Scikit.ML.PipelineHelper;
using Scikit.ML.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestQuestion
    {
        [TestMethod]
        public void TestQ_KMeamsDataFrame()
        {
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new Microsoft.ML.Runtime.Data.DataKind?[] { Microsoft.ML.Runtime.Data.DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Microsoft.ML.Trainers.KMeansPlusPlusClusterer());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 13));
        }

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
        public void TestQ_KMeamsAPI()
        {
            var iris = FileHelper.GetTestFile("iris.txt");

            var pipeline = new LearningPipeline();
            pipeline.Add(new Microsoft.ML.Data.TextLoader(iris).CreateFrom<IrisObservation>(separator: '\t'));
            pipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            pipeline.Add(new Microsoft.ML.Trainers.KMeansPlusPlusClusterer());
            var model = pipeline.Train<IrisObservation, IrisPrediction>();
            var obs = new IrisObservation() {
                Sepal_length = 3.3f,
                Sepal_width = 1.6f,
                Petal_length = 0.2f,
                Petal_width = 5.1f,
            };
            var predictions = model.Predict(obs);
            Assert.IsTrue(predictions.PredictedLabel != 0);
        }
    }
}