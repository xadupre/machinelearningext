// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using MlFood.ML.PipelineTransforms;
using MlFood.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineTransforms
    {
        [TestMethod]
        public void TestDescribeTransformCode()
        {
            var env = EnvHelper.NewTlcEnvironmentTest();
            var inputs = InputOutput.CreateInputs();
            var data = env.CreateStreamingDataView(inputs);
            var args = new DescribeTransform.Arguments() { columns = new[] { "X" } };
            var tr = new DescribeTransform(env, args, data);

            var values = new List<int>();
            using (var cursor = tr.GetRowCursor(i => true))
            {
                var columnGetter = cursor.GetGetter<DvInt4>(1);
                while (cursor.MoveNext())
                {
                    DvInt4 got = 0;
                    columnGetter(ref got);
                    values.Add((int)got);
                }
            }
            Assert.AreEqual(values.Count, 4);
        }

        [TestMethod]
        public void TestDescribeTransformSaveDataAndZip()
        {
            var env = EnvHelper.NewTlcEnvironmentTest();
            var inputs = InputOutput.CreateInputs();
            var data = env.CreateStreamingDataView(inputs);
            var args = new DescribeTransform.Arguments() { columns = new[] { "X" } };
            var tr = new DescribeTransform(env, args, data);

            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;

            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var saver = env.CreateSaver("Text");
            using (var fs2 = File.Create(outputDataFilePath))
                saver.SaveData(fs2, tr, 0);
            Assert.IsTrue(File.Exists(outputDataFilePath));

            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            using (var ch = env.Start("SaveModel"))
            using (var fs = File.Create(outModelFilePath))
            {
                var trainingExamples = env.CreateExamples(tr, null);
                TrainUtils.SaveModel(env, ch, fs, null, trainingExamples);
            }
            Assert.IsTrue(File.Exists(outModelFilePath));

            var outputDataFilePath2 = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var deserializedData = env.LoadTransforms(fs, data);
                var saver2 = env.CreateSaver("Text");
                var columns = new int[deserializedData.Schema.ColumnCount];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = i;
                using (var fs2 = File.Create(outputDataFilePath2))
                    saver2.SaveData(fs2, deserializedData, columns);
            }

            var d1 = File.ReadAllText(outputDataFilePath);
            Assert.IsTrue(d1.Length > 0);
            var d2 = File.ReadAllText(outputDataFilePath2);
            Assert.AreEqual(d1, d2);
        }
    }
}
