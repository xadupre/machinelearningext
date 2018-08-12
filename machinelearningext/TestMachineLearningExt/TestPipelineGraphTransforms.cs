// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineGraphTransforms
    {
        #region Chain, SelectTag, Tagged

        [TestMethod]
        public void TestTagViewTransform()
        {
            var host = EnvHelper.NewTestEnvironment();

            var inputs = new[] {
                new ExampleA() { X = new float[] { 0, 1 } },
                new ExampleA() { X = new float[] { 2, 3 } }
            };

            IDataView loader = host.CreateStreamingDataView(inputs);
            var data = host.CreateTransform("Scaler{col=X1:X}", loader);
            data = host.CreateTransform("tag{t=memory}", data);

            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
            TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
        }

        [TestMethod]
        public void TestSelectTagContactViewTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var firstData = FileHelper.GetOutputFile("first.idv", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();

            var inputs = new[] {
                new ExampleA() { X = new float[] { 0, 1, 4 } },
                new ExampleA() { X = new float[] { 2, 3, 7 } }
            };

            // Create IDV
            IDataView loader = env.CreateStreamingDataView(inputs);
            var saver = ComponentCreation.CreateSaver(env, "binary");
            using (var ch = env.Start("save"))
            {
                using (var fs0 = env.CreateOutputFile(firstData))
                    DataSaverUtils.SaveDataView(ch, saver, loader, fs0, true);

                // Create parallel pipeline
                loader = env.CreateStreamingDataView(inputs);
                var data = env.CreateTransform("Scaler{col=X1:X}", loader);
                data = env.CreateTransform(string.Format("selecttag{{t=first s=second f={0}}}", firstData), data);
                data = env.CreateTransform("Scaler{col=X1:X}", data);
                var merged = env.CreateTransform("append{t=first}", data);

                // Save the outcome
                var text = env.CreateSaver("Text");
                var columns = new int[merged.Schema.ColumnCount];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = i;
                using (var fs2 = File.Create(outData))
                    text.SaveData(fs2, merged, columns);

                // Final checking
                var lines = File.ReadAllLines(outData);
                if (!lines.Any())
                    throw new Exception("Empty file.");
                if (lines.Length != 11)
                    throw new Exception("Some lines are missing.");
                ch.Done();
            }
        }

        [TestMethod]
        public void TestChainTransform()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var host = EnvHelper.NewTestEnvironment();
            IDataView loader = host.CreateStreamingDataView(inputs);
            var chained = host.CreateTransform("ChainTrans{ xf1=Scaler{col=X2:X} xf2=Scaler{col=X3:X2} }", loader);
            var schStr = SchemaHelper.ToString(chained.Schema);
            Assert.AreEqual(schStr, "X:Vec<R4,3>:0; X2:Vec<R4,3>:1; X3:Vec<R4,3>:2");

            using (var cursor = chained.GetRowCursor(i => true))
            {
                var outValues0 = new List<float[]>();
                var outValues1 = new List<float[]>();
                var outValues2 = new List<float[]>();
                var colGetter0 = cursor.GetGetter<VBuffer<float>>(0);
                var colGetter1 = cursor.GetGetter<VBuffer<float>>(1);
                var colGetter2 = cursor.GetGetter<VBuffer<float>>(2);
                while (cursor.MoveNext())
                {
                    VBuffer<float> got = new VBuffer<float>();
                    colGetter0(ref got);
                    outValues0.Add(got.DenseValues().ToArray());
                    colGetter1(ref got);
                    outValues1.Add(got.DenseValues().ToArray());
                    colGetter2(ref got);
                    outValues2.Add(got.DenseValues().ToArray());
                }
                if (outValues0.Count != 2)
                    throw new Exception("expected 2");
                if (outValues1.Count != 2)
                    throw new Exception("expected 2");
                if (outValues2.Count != 2)
                    throw new Exception("expected 2");
                for (int i = 0; i < outValues0.Count; ++i)
                {
                    if (outValues0[i].Length != outValues1[i].Length)
                        throw new Exception("mismatch");
                    if (outValues2[i].Length != outValues1[i].Length)
                        throw new Exception("mismatch");
                    for (int j = 0; j < outValues0[i].Length; ++j)
                    {
                        if (outValues0[i][j] == outValues1[i][j])
                            throw new Exception("mismatch");
                        if (outValues2[i][j] != outValues1[i][j])
                            throw new Exception("mismatch");
                    }
                }
            }
        }

        [TestMethod]
        public void TestChainTransformSerialize()
        {
            var host = EnvHelper.NewTestEnvironment();

            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            IDataView loader = host.CreateStreamingDataView(inputs);
            IDataTransform data = host.CreateTransform("Scaler{col=X4:X}", loader);
            data = host.CreateTransform("ChainTrans{ xf1=Scaler{col=X2:X} xf2=Poly{col=X3:X2} }", data);

            // We create a specific folder in build/UnitTest which will contain the output.
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
            TestTransformHelper.SerializationTestTransform(host, outModelFilePath, data, loader, outData, outData2);
        }

        [TestMethod]
        public void TestChainTransformSerializeWithKMeans()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris_binary.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Features:R4:1-4 header=+}", 
                                new MultiFileSource(dataFilePath));

            var xf = env.CreateTransform("ChainTrans{xf1=Scaler{col=Features} xf2=Scaler{col=Features}}", loader);
            var roles = env.CreateExamples(xf, "Features");
            var trainer = env.CreateTrainer("KMeansPlusPlus{k=5}");
            using (var ch = env.Start("Train"))
            {
                var pred = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                     PredictionKind.Clustering, false);
            }
        }

        #endregion

        #region SplitTrainTest

        [TestMethod]
        [Ignore]
        public void TestTrainTestSinglePipelineWithTags()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));

            var pipe = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            pipe = env.CreateTransform("SplitTrainTest{col=base tag=train tag=test}", pipe);
            //pipe = env.CreateTransform("Tag{tag=unused2}", pipe); // just to check
            pipe = env.CreateTransform("SelectTag{tag=unused selectTag=train}", pipe);
            pipe = env.CreateTransform("cst{in=Features out=Ex:R4 code={O.Ex = I.Features[0] * I.Features[1];}}", pipe);
            pipe = env.CreateTransform("SelectTag{tag=newTrain selectTag=test}", pipe);
            pipe = env.CreateTransform("cst{in=Features out=Ex:R4 code={O.Ex = - I.Features[0] * I.Features[1];}}", pipe);
            pipe = env.CreateTransform("Append{tag=newTrain}", pipe);

            string schema = SchemaHelper.ToString(pipe.Schema);
            var cursor = pipe.GetRowCursor(i => true);
            string schema2 = SchemaHelper.ToString(cursor.Schema);
            if (schema != schema2)
                throw new Exception("Schema mismatch.");
            var getX = cursor.GetGetter<float>(0);
            var listx = new List<float>();
            float vx = 0;
            while (cursor.MoveNext())
            {
                getX(ref vx);
                listx.Add(vx);
            }
            if (listx.Count < 120)
                throw new Exception("Something is missing.");

            // The model is not serializable as it is because TLC only expects
            // to have one pipeline and no branches.
            // TestHelper.FinalizeSerializationTest(env, outModelFilePath, pipe, loader, outData, outData2);

            // Checks the outputs.
            var saver = env.CreateSaver("Text");
            var columns = new string[pipe.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = pipe.Schema.GetColumnName(i);
            using (var fs2 = File.Create(outData))
                saver.SaveData(fs2, pipe, StreamHelper.GetColumnsIndex(pipe.Schema));

            var lines = File.ReadAllLines(outData);
            if (lines.Length < 120)
                throw new Exception("Something is missing:" + string.Join("\n", lines));
        }

        [TestMethod]
        [Ignore]
        public void TrainTestSingleTrainingPipelineWithTags()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));

            var pipe = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            pipe = env.CreateTransform("SplitTrainTest{col=base tag=train tag=test}", pipe);
            pipe = env.CreateTransform("SelectTag{tag=unused selectTag=train}", pipe);
            pipe = env.CreateTransform(string.Format("TagTrainScore{{tag=trainP out={0} tr=mlr}}", outModelFilePath), pipe);
            pipe = env.CreateTransform("SelectTag{tag=scoredTrain selectTag=test}", pipe);
            pipe = env.CreateTransform("TagScore{in=trainP}", pipe);

            var cursor = pipe.GetRowCursor(i => true);
            string schema = SchemaHelper.ToString(pipe.Schema);
            string schema2 = SchemaHelper.ToString(cursor.Schema);
            if (schema != schema2)
                throw new Exception("Schema mismatch.");
            long count = DataViewUtils.ComputeRowCount(pipe);
            if (count != 49)
                throw new Exception(string.Format("Unexpected number of rows {0}", count));

            // Checks the outputs.
            var saver = env.CreateSaver("Text");
            var columns = new string[pipe.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = pipe.Schema.GetColumnName(i);
            using (var fs2 = File.Create(outData))
                saver.SaveData(fs2, pipe, StreamHelper.GetColumnsIndex(pipe.Schema));

            var lines = File.ReadAllLines(outData);
            if (lines.Length < 40)
                throw new Exception("Something is missing:" + string.Join("\n", lines));
            if (lines.Length > 70)
                throw new Exception("Too much data:" + string.Join("\n", lines));

            TestTransformHelper.SerializationTestTransform(env, outModelFilePath, pipe, loader, outData, outData2);
        }

        [TestMethod]
        [Ignore]
        public void TrainTestPipelinePredictTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("mc_iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));

            var pipe = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            pipe = env.CreateTransform("SplitTrainTest{col=base tag=train tag=test}", pipe);
            pipe = env.CreateTransform("SelectTag{tag=unused selectTag=train}", pipe);
            pipe = env.CreateTransform(string.Format("TagTrainScore{{tag=trainP out={0} tr=mlr}}", outModelFilePath), pipe);
            pipe = env.CreateTransform("SelectTag{tag=scoredTrain selectTag=test}", pipe);
            pipe = env.CreateTransform("Predict{in=trainP}", pipe);

            string schema = SchemaHelper.ToString(pipe.Schema);
            var cursor = pipe.GetRowCursor(i => true);
            string schema2 = SchemaHelper.ToString(cursor.Schema);
            if (schema != schema2)
                throw new Exception("Schema mismatch.");
            long count = DataViewUtils.ComputeRowCount(pipe);
            if (count != 49)
                throw new Exception(string.Format("Unexpected number of rows {0}", count));

            // Checks the outputs.
            var saver = env.CreateSaver("Text");
            var columns = new string[pipe.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = pipe.Schema.GetColumnName(i);
            using (var fs2 = File.Create(outData))
                saver.SaveData(fs2, pipe, StreamHelper.GetColumnsIndex(pipe.Schema));

            var lines = File.ReadAllLines(outData);
            if (lines.Length < 40)
                throw new Exception("Something is missing:" + string.Join("\n", lines));
            if (lines.Length > 70)
                throw new Exception("Too much data:" + string.Join("\n", lines));

            TestTransformHelper.SerializationTestTransform(env, outModelFilePath, pipe, loader, outData, outData2);
        }

        #endregion
    }
}
