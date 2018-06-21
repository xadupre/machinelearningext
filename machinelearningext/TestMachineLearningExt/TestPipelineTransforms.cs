// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Ext.PipelineTransforms;
using Microsoft.ML.Ext.TestHelper;


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
}
