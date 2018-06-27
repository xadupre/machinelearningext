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
        public void TestDataFrameOpMult()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] * df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 0f);
            Assert.AreEqual(df.iloc[1, 3], 1.1f);

            df["AA2"] = df["AA"] * 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], (DvInt4)0);
            Assert.AreEqual(df.iloc[1, 4], (DvInt4)10);
        }

        [TestMethod]
        public void TestDataFrameOpMinus()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] - df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], -1f);
            Assert.AreEqual(df.iloc[1, 3], 1 - 1.1f);

            df["AA2"] = df["AA"] - 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], (DvInt4)(-10));
            Assert.AreEqual(df.iloc[1, 4], (DvInt4)(-9));
        }

        [TestMethod]
        public void TestDataFrameOpEqual()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] == df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], DvBool.False);
            Assert.AreEqual(df.iloc[1, 3], DvBool.False);

            df["AA2"] = df["AA"] == 0;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], DvBool.True);
            Assert.AreEqual(df.iloc[1, 4], DvBool.False);

            var view = df[df["AA"] == 0];
            Assert.AreEqual(view.Shape, new Tuple<int, int>(1, 5));
        }

        [TestMethod]
        public void TestDataFrameOpDiv()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            Assert.AreEqual(text, tos);
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 3));

            df["BB*BB"] = df["AA"] / df["BB"];
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 4));
            Assert.AreEqual(df.iloc[0, 3], 0f);
            Assert.AreEqual(df.iloc[1, 3], 1 / 1.1f);

            df["AA2"] = df["AA"] / 10;
            Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 5));
            Assert.AreEqual(df.iloc[0, 4], (DvInt4)(0));
            Assert.AreEqual(df.iloc[1, 4], (DvInt4)(0));
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
            df.iloc[df["AA"].Filter<DvInt4>(c => (int)c == 1), 2] = "changed";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed");
            df.loc[df["AA"].Filter<DvInt4>(c => (int)c == 1), "CC"] = "changed2";
            Assert.AreEqual(df.iloc[1, 2].ToString(), "changed2");
        }

        [TestMethod]
        public void TestDataFrameOperationCopy()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
            var df = DataFrame.ReadStr(text);
            var tos = df.ToString();
            var copy = df.Copy();
            var tos2 = copy.ToString();
            Assert.AreEqual(tos, tos2);
            copy.iloc[copy["AA"].Filter<DvInt4>(c => (int)c == 1), 2] = "changed";
            tos2 = copy.ToString();
            Assert.AreNotEqual(tos, tos2);
        }

        [TestMethod]
        public void TestDataViewFrame()
        {
            var env = EnvHelper.NewTestEnvironment();
            var text = "AA,BB,CC\n0,1,text\n1,1.1,text2\n2,2.1,text3";
            var df = DataFrame.ReadStr(text);
            var view = df[new int[] { 0, 2 }, new int[] { 0, 2 }];
            var tv = view.ToString();
            Assert.AreEqual("AA,CC\n0,text\n2,text3", tv);
            var dfview = view.Copy();
            var tv2 = dfview.ToString();
            Assert.AreEqual("AA,CC\n0,text\n2,text3", tv2);
            dfview["AA1"] = view["AA"] + 1;
            var tv3 = dfview.ToString();
            Assert.AreEqual("AA,CC,AA1\n0,text,1\n2,text3,3", tv3);
            var view2 = df[df.ALL, new[] { "AA", "CC" }];
            var tv4 = view2.ToString();
            Assert.AreEqual("AA,CC\n0,text\n1,text2\n2,text3", tv4);
            var view3 = df[new[] { 0 }, df.ALL];
            var tv5 = view3.ToString();
            Assert.AreEqual("AA,BB,CC\n0,1,text", tv5);
        }

        [TestMethod]
        public void TestCreateFromArrays()
        {
            var df = new DataFrame();
            df.AddColumn("i", new int[] { 0, 1 });
            df.AddColumn("x", new float[] { 0.5f, 1.5f });
            var tx = df.ToString();
            Assert.AreEqual(tx, "i,x\n0,0.5\n1,1.5");
        }
    }
}

