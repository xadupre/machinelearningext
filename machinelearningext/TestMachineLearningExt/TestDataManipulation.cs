// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.DataManipulation;
using Microsoft.ML.Ext.TestHelper;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Ext.PipelineHelper;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestDataManipulation
    {
        [TestMethod]
        public void TestReadCsvSimple()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t');
            Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 5));
            var sch = df.Schema;
            Assert.AreEqual(sch.GetColumnName(0), "Label");
            Assert.AreEqual(sch.GetColumnName(1), "Sepal_length");
            Assert.AreEqual(sch.GetColumnType(0), NumberType.I4);
            Assert.AreEqual(sch.GetColumnType(1), NumberType.R4);
            Assert.AreEqual(df.iloc[0, 0], (DvInt4)0);
            Assert.AreEqual(df.iloc[1, 0], (DvInt4)0);
            Assert.AreEqual(df.iloc[140, 0], (DvInt4)2);
            df.iloc[1, 0] = (DvInt4)10;
            Assert.AreEqual(df.iloc[1, 0], (DvInt4)10);
            df.loc[1, "Label"] = (DvInt4)11;
            Assert.AreEqual(df.loc[1, "Label"], (DvInt4)11);
            var d = df[1];
            Assert.AreEqual(d.Count, 5);
            Assert.AreEqual(d["Label"], (DvInt4)11);
            var col = df["Label"];
            Assert.AreEqual(col.Length, 150);
            df["Label2"] = df["Label"];
            col = df["Label2"];
            Assert.AreEqual(col.Length, 150);
            Assert.AreEqual(df.loc[1, "Label2"], (DvInt4)11);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 6));
        }

        [TestMethod]
        public void TestReadView()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrame.ReadCsvToTextLoader(iris, sep: '\t', host: env.Register("TextLoader"));
            var df = DataFrame.ReadView(loader);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(150, 5));
            var sch = df.Schema;
            Assert.AreEqual(sch.GetColumnName(0), "Label");
            Assert.AreEqual(sch.GetColumnName(1), "Sepal_length");
            Assert.AreEqual(sch.GetColumnType(0), NumberType.I4);
            Assert.AreEqual(sch.GetColumnType(1), NumberType.R4);
            Assert.AreEqual(df.iloc[0, 0], (DvInt4)0);
            Assert.AreEqual(df.iloc[1, 0], (DvInt4)0);
            Assert.AreEqual(df.iloc[140, 0], (DvInt4)2);
        }

        [TestMethod]
        public void TestReadViewEqual()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrame.ReadCsvToTextLoader(iris, sep: '\t');
            var df1 = DataFrame.ReadCsv(iris, sep: '\t');
            var df2 = DataFrame.ReadView(loader);
            Assert.IsTrue(df1 == df2);
            df2.iloc[1, 0] = (DvInt4)10;
            Assert.IsTrue(df1 != df2);
        }

        [TestMethod]
        public void TestReadTextLoaderSimple()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrame.ReadCsvToTextLoader(iris, sep: '\t');
            var sch = loader.Schema;
            Assert.AreEqual(sch.ColumnCount, 5);
            Assert.AreEqual(sch.GetColumnName(0), "Label");
            Assert.AreEqual(sch.GetColumnName(1), "Sepal_length");
        }

        [TestMethod]
        public void TestReadToCsv()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t');
            var outfile = FileHelper.GetOutputFile("iris_copy.txt", methodName);
            df.ToCsv(outfile);
            Assert.IsTrue(File.Exists(outfile));
        }

        [TestMethod]
        public void TestReadStr()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var df1 = DataFrame.ReadCsv(iris, sep: '\t');
            var content = File.ReadAllText(iris);
            var df2 = DataFrame.ReadStr(content, sep: '\t');
            Assert.IsTrue(df1 == df2);
        }

        [TestMethod]
        public void TestDataFrameScoringMulti()
        {
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });
            var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
            var trainingData = env.CreateExamples(conc, "Feature", label: "Label");
            var trainer = env.CreateTrainer("ova{p=lr}");
            using (var ch = env.Start("test"))
            {
                var pred = trainer.Train(env, ch, trainingData);
                var scorer = trainer.GetScorer(pred, trainingData, env, null);
                var predictions = DataFrame.ReadView(scorer);
                var v = predictions.iloc[0, 7];
                Assert.AreEqual(v, (uint)1);
                Assert.AreEqual(predictions.Schema.GetColumnName(5), "Feature.0");
                Assert.AreEqual(predictions.Schema.GetColumnName(6), "Feature.1");
                Assert.AreEqual(predictions.Schema.GetColumnName(7), "PredictedLabel");
                Assert.AreEqual(predictions.Shape, new Tuple<int, int>(150, 11));
            }
        }

        [TestMethod]
        public void TestDataFrameScoringBinary()
        {
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });
            var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
            var trainingData = env.CreateExamples(conc, "Feature", label: "Label");
            var trainer = env.CreateTrainer("lr");
            using (var ch = env.Start("test"))
            {
                var pred = trainer.Train(env, ch, trainingData);
                var scorer = trainer.GetScorer(pred, trainingData, env, null);
                var predictions = DataFrame.ReadView(scorer);
                var v = predictions.iloc[0, 7];
                Assert.AreEqual(v, DvBool.False);
                Assert.AreEqual(predictions.Schema.GetColumnName(5), "Feature.0");
                Assert.AreEqual(predictions.Schema.GetColumnName(6), "Feature.1");
                Assert.AreEqual(predictions.Schema.GetColumnName(7), "PredictedLabel");
                Assert.AreEqual(predictions.Shape, new Tuple<int, int>(150, 10));
            }
        }

        [TestMethod]
        public void TestDataFrameScoringMultiEntryPoints2()
        {
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new StochasticDualCoordinateAscentRegressor());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 8));
        }

        [TestMethod]
        public void TestDataFrameOperation()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BBxBB"] = df["AA"] + df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 1f);
            Assert.AreEqual(df.iloc[1, 3], 2.1f);

            df["BBxBB2"] = df["BB"] + df["AA"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], 1f);
            Assert.AreEqual(df.iloc[1, 4], 2.1f);

            df["AA2"] = df["AA"] + 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 6));
            Assert.AreEqual(df.iloc[0, 5], (DvInt4)10);
            Assert.AreEqual(df.iloc[1, 5], (DvInt4)11);

            df["CC2"] = df["CC"] + "10";
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 7));
            Assert.AreEqual(df.iloc[0, 6].ToString(), "text10");
            Assert.AreEqual(df.iloc[1, 6].ToString(), "text210");
        }

        [TestMethod]
        public void TestDataFrameOperationSet()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            df.loc["CC"] = "changed";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed");
        }

        [TestMethod]
        public void TestDataFrameOperationIEnumerable()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));
            df[df["AA"].Filter<DvInt4>(c => (int)c == 1), 2] = "changed";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed");
            df[df["AA"].Filter<DvInt4>(c => (int)c == 1), "CC"] = "changed2";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed2");
        }
    }
}
