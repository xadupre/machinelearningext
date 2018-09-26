// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.TextAnalytics;
using Scikit.ML.TestHelper;
using Scikit.ML.ScikitAPI;
using Scikit.ML.DataManipulation;
using Scikit.ML.ProductionPrediction;


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
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host);
                var predictor = pipe.Train(data);
                Assert.IsTrue(predictor != null);
                var data2 = host.CreateStreamingDataView(inputs2);
                var predictions = pipe.Transform(data2);
                var df = DataFrame.ReadView(predictions);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                Assert.AreEqual(dfs2, "X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8;-1,-10,-100,1,10,100,100,1000,10000;-2,-3,-5,4,6,10,9,15,25");
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
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host);
                var predictor = pipe.Train(data);
                Assert.IsTrue(predictor != null);
                var data2 = host.CreateStreamingDataView(inputs2);
                var predictions = pipe.Transform(data2);
                var df = DataFrame.ReadView(predictions);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                expected = dfs2;
                Assert.AreEqual(dfs2, "X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8;-1,-10,-100,1,10,100,100,1000,10000;-2,-3,-5,4,6,10,9,15,25");
                pipe.Save(output);
            }
            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data2 = host.CreateStreamingDataView(inputs2);
                var pipe2 = new ScikitPipeline(output, host);
                var predictions = pipe2.Transform(data2);
                var df = DataFrame.ReadView(predictions);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(2, 9));
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                Assert.AreEqual(expected, dfs2);
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
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host);
                var predictor = pipe.Train(data, feature: "X");
                Assert.IsTrue(predictor != null);
                var data2 = host.CreateStreamingDataView(inputs2);
                var predictions = pipe.Predict(data2);
                var df = DataFrame.ReadView(predictions);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                Assert.IsTrue(dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"));
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
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host);
                var predictor = pipe.Train(data, feature: "X");
                Assert.IsTrue(predictor != null);
                var data2 = host.CreateStreamingDataView(inputs2);
                var predictions = pipe.Predict(data2);
                var df = DataFrame.ReadView(predictions);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                Assert.IsTrue(dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"));
                expected = dfs2;
                pipe.Save(output);
            }
            using (var host = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var data2 = host.CreateStreamingDataView(inputs2);
                var pipe2 = new ScikitPipeline(output, host);
                var predictions = pipe2.Predict(data2);
                var df = DataFrame.ReadView(predictions);
                Assert.AreEqual(df.Shape, new Tuple<int, int>(4, 12));
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                Assert.AreEqual(expected, dfs2);
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
            ILogWriter logout = new LogWriter(s => stdout.Add(s));
            ILogWriter logerr = new LogWriter(s => stderr.Add(s));

            using (var host = new DelegateEnvironment(conc: 1, outWriter: logout, errWriter: logerr, verbose: 1))
            {
                ComponentHelper.AddStandardComponents(host);
                var data = host.CreateStreamingDataView(inputs);
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, host: host);
                var predictor = pipe.Train(data);
                Assert.IsTrue(predictor != null);
            }
            Assert.IsTrue(stdout.Count > 0);
            Assert.AreEqual(stderr.Count, 0);
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
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host: host);
                var predictor = pipe.Train(data, feature: "X");
                Assert.IsTrue(predictor != null);
            }
            Assert.AreEqual(stdout.Count, 0);
            Assert.AreEqual(stderr.Count, 0);
        }

        private IDataScorerTransform _TrainSentiment()
        {
            bool normalize = true;

            var args = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };

            var args2 = new TextTransform.Arguments()
            {
                Column = new TextTransform.Column
                {
                    Name = "Features",
                    Source = new[] { "SentimentText" }
                },
                KeepDiacritics = false,
                KeepPunctuations = false,
                TextCase = TextNormalizerTransform.CaseNormalizationMode.Lower,
                OutputTokens = true,
                StopWordsRemover = new PredefinedStopWordsRemoverFactory(),
                VectorNormalizer = normalize ? TextTransform.TextNormKind.L2 : TextTransform.TextNormKind.None,
                CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false },
                WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 2, AllLengths = true },
            };

            var trainFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-data.tsv");

            using (var env = EnvHelper.NewTestEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = TextLoader.ReadFile(env, args, new MultiFileSource(trainFilename));

                var trans = TextTransform.Create(env, args2, loader);

                // Train
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1
                });

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var predictor = trainer.Train(new Microsoft.ML.Runtime.TrainContext(trainRoles));

                var scoreRoles = new RoleMappedData(trans, label: "Label", feature: "Features");
                return ScoreUtils.GetScorer(predictor, scoreRoles, env, trainRoles.Schema);
            }
        }

        private TimeSpan _MeasureTime(int conc, int engine, IDataScorerTransform scorer)
        {
            var args = new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.BL, 0),
                    new TextLoader.Column("SentimentText", DataKind.Text, 1)
                }
            };

            var testFilename = FileHelper.GetTestFile("wikipedia-detox-250-line-test.tsv");

            using (var env = EnvHelper.NewTestEnvironment(seed: 1, conc: conc))
            {

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = TextLoader.ReadFile(env, args, new MultiFileSource(testFilename));
                var testData = testLoader.AsEnumerable<SentimentData>(env, false);

                if (engine == 1)
                {
                    var model = env.CreatePredictionEngine<SentimentData, SentimentPrediction>(scorer);
                    var sw = new Stopwatch();
                    sw.Start();
#if(DEBUG)
                    int N = 1;
#else
                    int N = 100;
#endif
                    for (int i = 0; i < N; ++i)
                        foreach (var input in testData)
                            model.Predict(input);
                    sw.Stop();
                    return sw.Elapsed;
                }
                else if (engine == 2)
                {
                    /*
                    var model = new ValueMapperPredictionEngine<SentimentData>(env, scorer);
                    var sw = new Stopwatch();
                    sw.Start();
                    for (int i = 0; i < 100; ++i)
                        foreach (var input in testData)
                            model.Predict(input);
                    sw.Stop();
                    return sw.Elapsed;
                    */
                }
            }
            throw new NotImplementedException();
        }

        [TestMethod]
        public void TestScikitAPI_EngineSimpleTrainAndPredict()
        {
            var scorer = _TrainSentiment();
            var time1 = new TimeSpan[4];
            var time2 = new TimeSpan[4];
            for (int i = 1; i <= 4; ++i)
            {
                time1[i - 1] = _MeasureTime(i, 1, scorer);
            }
        }
    }
}
