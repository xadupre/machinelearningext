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
using Scikit.ML.ProductionPrediction;
using Scikit.ML.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class Test_Benchmarks
    {

        [TestMethod]
        public void TestValueMapperPredictionEngineMultiThread()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");

            using (var env = EnvHelper.NewTestEnvironment())
            {
                using (var engine0 = new ValueMapperPredictionEngineFloat(env, name, getterEachTime: true, conc: 1))
                {
                    var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
                    var exp = new float[100];
                    for (int i = 0; i < exp.Length; ++i)
                    {
                        feat[0] = i;
                        exp[i] = engine0.Predict(feat);
                        Assert.IsFalse(float.IsNaN(exp[i]));
                        Assert.IsFalse(float.IsInfinity(exp[i]));
                    }

                    var dico = new Dictionary<Tuple<bool, int>, TimeSpan>();

                    foreach (var each in new[] { false, true })
                    {
                        foreach (int th in new int[] { 2, 0, 1, 3 })
                        {
                            var engine = new ValueMapperPredictionEngineFloat(env, name, getterEachTime: each, conc: th);
                            var sw = new Stopwatch();
                            sw.Start();
                            for (int i = 0; i < exp.Length; ++i)
                            {
                                feat[0] = i;
                                var res = engine.Predict(feat);
                                Assert.AreEqual(exp[i], res);
                            }
                            sw.Stop();
                            dico[new Tuple<bool, int>(each, th)] = sw.Elapsed;
                        }
                    }
                    Assert.AreEqual(dico.Count, 8);
                }
            }
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
