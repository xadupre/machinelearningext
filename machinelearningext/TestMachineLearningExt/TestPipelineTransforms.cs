// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.TestHelper;
using Legacy = Microsoft.ML.Legacy;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineTransforms
    {
        #region DescribeTransform

        [TestMethod]
        public void TestI_DescribeTransformCode()
        {
            using (var env = EnvHelper.NewTestEnvironment())
            {
                var inputs = InputOutput.CreateInputs();
                var data = env.CreateStreamingDataView(inputs);
                var args = new DescribeTransform.Arguments() { columns = new[] { "X" } };
                var tr = new DescribeTransform(env, args, data);

                var values = new List<int>();
                using (var cursor = tr.GetRowCursor(i => true))
                {
                    var columnGetter = cursor.GetGetter<int>(1);
                    while (cursor.MoveNext())
                    {
                        int got = 0;
                        columnGetter(ref got);
                        values.Add((int)got);
                    }
                }
                Assert.AreEqual(values.Count, 4);
            }
        }

        [TestMethod]
        public void TestI_DescribeTransformSaveDataAndZip()
        {
            using (var env = EnvHelper.NewTestEnvironment())
            {
                var inputs = InputOutput.CreateInputs();
                var data = env.CreateStreamingDataView(inputs);
                var args = new DescribeTransform.Arguments() { columns = new[] { "X" } };
                var tr = new DescribeTransform(env, args, data);

                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;

                var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
                StreamHelper.SavePredictions(env, tr, outputDataFilePath);
                Assert.IsTrue(File.Exists(outputDataFilePath));

                var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
                StreamHelper.SaveModel(env, tr, outModelFilePath);
                Assert.IsTrue(File.Exists(outModelFilePath));

                var outputDataFilePath2 = FileHelper.GetOutputFile("outputDataFilePath2.txt", methodName);
                StreamHelper.SavePredictions(env, outModelFilePath, outputDataFilePath2, data);
                Assert.IsTrue(File.Exists(outputDataFilePath2));

                var d1 = File.ReadAllText(outputDataFilePath);
                Assert.IsTrue(d1.Length > 0);
                var d2 = File.ReadAllText(outputDataFilePath2);
                Assert.AreEqual(d1, d2);
            }
        }

        #endregion

        #region PassThroughTransform

        [TestMethod]
        public void TestI_PassThroughTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
            var tempFile = FileHelper.GetOutputFile("dump.idv", methodName);

            using (var env = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 col=Uid:TX:5 header=+}",
                    new MultiFileSource(dataFilePath));

                var xf1 = env.CreateTransform("Concat{col=Feat:Slength,Swidth}", loader);
                var xf2 = env.CreateTransform("Scaler{col=Feat}", xf1);
                var xf3 = env.CreateTransform(string.Format("DumpView{{s=+ f={0}}}", tempFile), xf2);
                TestTransformHelper.SerializationTestTransform(env, outModelFilePath, xf3, loader, outData, outData2, false);
                if (!File.Exists(tempFile))
                    throw new FileNotFoundException(tempFile);
            }
        }

        [TestMethod]
        public void TestEP_PassThroughTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var iris = FileHelper.GetTestFile("iris.txt");
            var outPass = FileHelper.GetOutputFile("data.idv", methodName);
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new Legacy.Transforms.ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.Scaler("Features"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.PassThrough() { Filename = outPass, SaveOnDisk = true });
            learningPipeline.Add(new Legacy.Trainers.StochasticDualCoordinateAscentRegressor());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(new Tuple<int, int>(150, 8), dfout.Shape);
            Assert.IsTrue(File.Exists(outPass));
        }

        #endregion

        #region ULabel2Float

        [TestMethod]
        public void TestI_ULabelToR4LabelTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris_binary.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            using (var env = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var loader = env.CreateLoader("Text{col=LabelText:TX:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                    new MultiFileSource(dataFilePath));

                var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
                var labelTx = env.CreateTransform("TermTransform{col=LabelU4:LabelText}", concat);
                var labelR4 = env.CreateTransform("U2R4{col=Label:LabelU4}", labelTx);
                var roles = env.CreateExamples(labelR4, "Features", "Label");
                var trainer = env.CreateTrainer("lr");
                using (var ch = env.Start("test"))
                {
                    var pred = trainer.Train(env, ch, roles);
                    TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                                trainer.Trainer.PredictionKind, true, ratio: 0.8f);
                    ch.Done();
                }
            }
        }

        [TestMethod]
        public void TestEP_ULabelToR4LabelTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new Legacy.Transforms.ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.Scaler("Features"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.ULabelToR4Label("Label"));
            learningPipeline.Add(new Legacy.Trainers.StochasticDualCoordinateAscentRegressor());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(new Tuple<int, int>(150, 8), dfout.Shape);
        }

        #endregion

        #region SortInDataFrameTransform, ExtendCacheTransform

        [TestMethod]
        public void TestSortInDataFrameTransformSimple()
        {
            var host = EnvHelper.NewTestEnvironment();

            var inputs = new InputOutput[] {
                new InputOutput() { X = new float[] { 0, 1 }, Y = 1 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 0 }
            };

            var data = host.CreateStreamingDataView(inputs);

            using (var cursor = data.GetRowCursor(i => true))
            {
                var sortedValues = new List<int>();
                var sortColumnGetter = cursor.GetGetter<int>(1);
                while (cursor.MoveNext())
                {
                    int got = 0;
                    sortColumnGetter(ref got);
                    sortedValues.Add((int)got);
                }
                if (sortedValues.Count != 2)
                    throw new Exception();
                if (sortedValues[0] != 1)
                    throw new Exception();
                if (sortedValues[1] != 0)
                    throw new Exception();
            }

            var args = new SortInDataFrameTransform.Arguments { sortColumn = "Y" };
            var transformedData = new SortInDataFrameTransform(host, args, data);
            var sorted = transformedData;
            LambdaTransform.CreateMap<InputOutput, InputOutput>(host, data,
                (input, output) =>
                {
                    output.X = input.X;
                    output.Y = input.Y;
                });

            using (var cursor = sorted.GetRowCursor(i => true))
            {
                var sortedValues = new List<int>();
                var sortColumnGetter = cursor.GetGetter<int>(1);
                while (cursor.MoveNext())
                {
                    int got = 0;
                    sortColumnGetter(ref got);
                    sortedValues.Add((int)got);
                }
                if (sortedValues.Count != 2)
                    throw new Exception();
                if (sortedValues[0] != 0)
                    throw new Exception();
                if (sortedValues[1] != 1)
                    throw new Exception();
            }
        }

        static void TestCacheTransformSimple(int nt, bool async)
        {
            var host = EnvHelper.NewTestEnvironment(conc: nt == 1 ? 1 : 0);

            var inputs = new InputOutput[] {
                new InputOutput() { X = new float[] { 0, 1 }, Y = 1 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 0 }
            };

            var data = host.CreateStreamingDataView(inputs);

            using (var cursor = data.GetRowCursor(i => true))
            {
                var sortedValues = new List<int>();
                var sortColumnGetter = cursor.GetGetter<int>(1);
                while (cursor.MoveNext())
                {
                    int got = 0;
                    sortColumnGetter(ref got);
                    sortedValues.Add((int)got);
                }
                if (sortedValues.Count != 2)
                    throw new Exception();
                if (sortedValues[0] != 1)
                    throw new Exception();
                if (sortedValues[1] != 0)
                    throw new Exception();
            }

            var args = new ExtendedCacheTransform.Arguments { numTheads = nt, async = async };
            var transformedData = new ExtendedCacheTransform(host, args, data);
            var lastTransform = transformedData;
            LambdaTransform.CreateMap<InputOutput, InputOutput>(host, data,
                (input, output) =>
                {
                    output.X = input.X;
                    output.Y = input.Y;
                });

            using (var cursor = lastTransform.GetRowCursor(i => true))
            {
                var sortedValues = new List<int>();
                var sortColumnGetter = cursor.GetGetter<int>(1);
                while (cursor.MoveNext())
                {
                    int got = 0;
                    sortColumnGetter(ref got);
                    sortedValues.Add((int)got);
                }
                if (sortedValues.Count != 2)
                    throw new Exception();
            }
        }

        [TestMethod]
        public void Testl_TransCacheTransformSimple()
        {
            TestCacheTransformSimple(1, false);
        }

        [TestMethod]
        public void Testl_TransCacheTransformSimpleAsync()
        {
            TestCacheTransformSimple(1, true);
            TestCacheTransformSimple(2, true);
        }

        [TestMethod]
        [Ignore()]  // Dataframe cannot yet be filled with multiple threads.
        public void Testl_TransCacheTransformSimpleNT()
        {
            TestCacheTransformSimple(2, false);
        }

        [TestMethod]
        public void TestDataViewCacheDataFrameSerialization()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));
            var sorted = env.CreateTransform("cachedf", loader);
            StreamHelper.SaveModel(env, sorted, outModelFilePath);

            using (var fs = File.OpenRead(outModelFilePath))
            {
                var deserializedData = env.LoadTransforms(fs, loader);
                var saver = env.CreateSaver("Text");
                using (var fs2 = File.Create(outputDataFilePath))
                    saver.SaveData(fs2, deserializedData,
                                   StreamHelper.GetColumnsIndex(deserializedData.Schema,
                                                                new[] { "Label", "Slength", "Swidth", "Plength", "Pwidth" }));
            }
        }

        [TestMethod]
        public void Testl_SortInMemoryShuffle()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("shuffled_iris.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=- sep=,}",
                new MultiFileSource(dataFilePath));
            var sorted = env.CreateTransform("sortmem{col=Label}", loader);

            var saver = env.CreateSaver("Text");
            using (var fs2 = File.Create(outputDataFilePath))
                saver.SaveData(fs2, sorted, StreamHelper.GetColumnsIndex(sorted.Schema, new[] { "Label", "Slength", "Swidth", "Plength", "Pwidth" }));

            var lines = File.ReadAllLines(outputDataFilePath);
            int begin = 0;
            for (; begin < lines.Length; ++begin)
            {
                if (lines[begin].StartsWith("Label"))
                    break;
            }
            lines = lines.Skip(begin).ToArray();
            var linesSorted = lines.OrderBy(c => c).ToArray();
            for (int i = 1; i < linesSorted.Length; ++i)
            {
                if (linesSorted[i - 1][0] > linesSorted[i][0])
                    throw new Exception("The output is not sorted.");
            }
        }

        [TestMethod]
        public void TestDataViewCacheDataFrameSerializationCacheFile()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var cacheFile = FileHelper.GetOutputFile("cacheFile.idv", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));
            var sorted = env.CreateTransform(string.Format("cachedf{{r=+ df=- f={0}}}", cacheFile), loader);
            StreamHelper.SaveModel(env, sorted, outModelFilePath);

            using (var fs = File.OpenRead(outModelFilePath))
            {
                var deserializedData = env.LoadTransforms(fs, loader);
                var saver = env.CreateSaver("Text");
                using (var fs2 = File.Create(outputDataFilePath))
                    saver.SaveData(fs2, deserializedData,
                                   StreamHelper.GetColumnsIndex(deserializedData.Schema, new[] { "Label", "Slength", "Swidth", "Plength", "Pwidth" }));
            }

            if (!File.Exists(cacheFile))
                throw new FileNotFoundException(cacheFile);
        }

        #endregion
    }
}
