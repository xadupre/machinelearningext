// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Scikit.ML.TestHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestScikitAPI
    {
        [TestMethod]
        public void TestScikitAPI_SimpleTransform()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    Assert.IsTrue(predictor != null);
                    var data2 = host.CreateStreamingDataView(inputs2);
                    var predictions = pipe.Transform(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(dfs2, "X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8;-1,-10,-100,1,10,100,100,1000,10000;-2,-3,-5,4,6,10,9,15,25");
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_SimpleTransform_Load()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var output = FileHelper.GetOutputFile("model.zip", methodName);
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            string expected = null;
            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    Assert.IsTrue(predictor != null);
                    var data2 = host.CreateStreamingDataView(inputs2);
                    var predictions = pipe.Transform(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    expected = dfs2;
                    Assert.AreEqual(dfs2, "X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8;-1,-10,-100,1,10,100,100,1000,10000;-2,-3,-5,4,6,10,9,15,25");
                    pipe.Save(output);
                }
            }
            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data2 = host.CreateStreamingDataView(inputs2);
                using (var pipe2 = new ScikitPipeline(output, host))
                {
                    var predictions = pipe2.Transform(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(expected, dfs2);
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_SimplePredictor()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } },
                new ExampleA() { X = new float[] { 3, 4, 5 } },
                new ExampleA() { X = new float[] { 3, 4, 7 } },
            };

            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    Assert.IsTrue(predictor != null);
                    var data2 = host.CreateStreamingDataView(inputs2);
                    var predictions = pipe.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.IsTrue(dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"));
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_SimplePredictor_Load()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var output = FileHelper.GetOutputFile("model.zip", methodName);
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 5 } },
                new ExampleA() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } },
                new ExampleA() { X = new float[] { 3, 4, 5 } },
                new ExampleA() { X = new float[] { 3, 4, 7 } },
            };

            string expected = null;
            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    Assert.IsTrue(predictor != null);
                    var data2 = host.CreateStreamingDataView(inputs2);
                    var predictions = pipe.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    if (!dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"))
                        throw new Exception($"Wrong starts\n{dfs2}");
                    expected = dfs2;
                    pipe.Save(output);
                }
            }
            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data2 = host.CreateStreamingDataView(inputs2);
                using (var pipe2 = new ScikitPipeline(output, host))
                {
                    var predictions = pipe2.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    Assert.AreEqual(expected, dfs2);
                }
            }
        }

        [TestMethod]
        public void TestScikitAPI_DelegateEnvironment()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            var stdout = new List<string>();
            var stderr = new List<string>();
            ILogWriter logout = new LogWriter((string s) =>
            {
                stdout.Add(s);
            });
            ILogWriter logerr = new LogWriter((string s) =>
            {
                stderr.Add(s);
            });

            using (var host = new DelegateEnvironment(conc: 1, outWriter: logout, errWriter: logerr, verbose: 3))
            using (var ch = host.Start("Train Pipeline"))
            {
                ComponentHelper.AddStandardComponents(host);
                ch.Info(MessageSensitivity.All, "Polynomial");
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host))
                {
                    var predictor = pipe.Train(data);
                    if (predictor == null)
                        throw new Exception("Predictor is null");
                }
            }
            if (stdout.Count == 0)
                throw new Exception("stdout is empty.");
            if (stderr.Count != 0)
                throw new Exception($"stderr not empty\n{string.Join("\n", stderr)}");
        }

        [TestMethod]
        public void TestScikitAPI_DelegateEnvironmentVerbose0()
        {
            var inputs = new[] {
                new ExampleA() { X = new float[] { 1, 10, 100 } },
                new ExampleA() { X = new float[] { 2, 3, 5 } }
            };

            var inputs2 = new[] {
                new ExampleA() { X = new float[] { -1, -10, -100 } },
                new ExampleA() { X = new float[] { -2, -3, -5 } }
            };

            var stdout = new List<string>();
            var stderr = new List<string>();
            ILogWriter logout = new LogWriter(s => stdout.Add(s));
            ILogWriter logerr = new LogWriter(s => stderr.Add(s));

            using (var host = new DelegateEnvironment(conc: 1, outWriter: logout, errWriter: logerr, verbose: 0))
            {
                ComponentHelper.AddStandardComponents(host);
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host: host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    Assert.IsTrue(predictor != null);
                }
            }
            Assert.AreEqual(stdout.Count, 0);
            Assert.AreEqual(stderr.Count, 0);
        }
    }
}
