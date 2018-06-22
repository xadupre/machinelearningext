// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.DataManipulation;
using Microsoft.ML.Ext.TestHelper;
using Microsoft.ML.Runtime.Api;


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
            var df = DataFrame.ReadCsv(env.Register("DataFrame"), iris, sep: '\t');
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
        }

        [TestMethod]
        public void TestReadView()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrame.ReadCsvToTextLoader(env.Register("DataFrame"), iris, sep: '\t');
            var df = DataFrame.ReadView(env.Register("view"), loader);
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
            var loader = DataFrame.ReadCsvToTextLoader(env.Register("DataFrame"), iris, sep: '\t');
            var df1 = DataFrame.ReadCsv(env.Register("DataFrame"), iris, sep: '\t');
            var df2 = DataFrame.ReadView(env.Register("view"), loader);
            Assert.IsTrue(df1 == df2);
            df2.iloc[1, 0] = (DvInt4)10;
            Assert.IsTrue(df1 != df2);
        }

        [TestMethod]
        public void TestReadTextLoaderSimple()
        {
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var loader = DataFrame.ReadCsvToTextLoader(env.Register("DataFrame"), iris, sep: '\t');
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
            var df = DataFrame.ReadCsv(env.Register("DataFrame"), iris, sep: '\t');
            var outfile = FileHelper.GetOutputFile("iris_copy.txt", methodName);
            df.ToCsv(outfile);
            Assert.IsTrue(File.Exists(outfile));
        }

        [TestMethod]
        public void TestReadStr()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var env = EnvHelper.NewTestEnvironment();
            var iris = FileHelper.GetTestFile("iris.txt");
            var df1 = DataFrame.ReadCsv(env.Register("DataFrame"), iris, sep: '\t');
            var content = File.ReadAllText(iris);
            var df2 = DataFrame.ReadStr(env.Register("DataFrame"), content, sep: '\t');
            Assert.IsTrue(df1 == df2);
        }

        [TestMethod]
        public void TestDataFrameScoring()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(env.Register("DataFrame"), iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });
            var conc = env.CreateTransform("Concat{col=Feature:Sepal_length,Sepal_width}", df);
            var trainingData = env.CreateExamples(conc, "Feature", label: "Label");
            string loadName;
            var trainer = env.CreateTrainer("lr", out loadName);
            using (var ch = env.Start("test"))
            {
                var pred = TrainUtils.Train(env, ch, trainingData, trainer, loadName, null, null, 0, false);
                var scorer = ScoreUtils.GetScorer(pred, trainingData, env, null);
                var predictions = DataFrame.ReadView(env.Register("predictions"), scorer);
                var v = predictions.iloc[0, 7];
                Assert.AreEqual(v, DvBool.False);
                Assert.AreEqual(predictions.Schema.GetColumnName(5), "Feature.0");
                Assert.AreEqual(predictions.Schema.GetColumnName(6), "Feature.1");
                Assert.AreEqual(predictions.Schema.GetColumnName(7), "PredictedLabel");
                Assert.AreEqual(predictions.Shape, new Tuple<int, int>(150, 10));
            }
        }
    }
}
