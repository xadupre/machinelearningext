// See the LICENSE file in the project root for more information.

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
using Scikit.ML.PipelineTransforms;


namespace TestProfileBenchmark
{
    public static class Benchmark_PredictionEngine
    {
        private static IDataScorerTransform _TrainSentiment()
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

        private static List<Tuple<int, TimeSpan, int>> _MeasureTime(int conc, 
            bool getterEachTime, string engine, IDataScorerTransform scorer, int N, int ncall, bool cacheScikit)
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
            var times = new List<Tuple<int, TimeSpan, int>>();

            using (var env = EnvHelper.NewTestEnvironment(seed: 1, conc: conc))
            {

                // Take a couple examples out of the test data and run predictions on top.
                var testLoader = TextLoader.ReadFile(env, args, new MultiFileSource(testFilename));
                IDataView cache;
                if (cacheScikit)
                    cache = new ExtendedCacheTransform(env, new ExtendedCacheTransform.Arguments(), testLoader);
                else
                    cache = new CacheDataView(env, testLoader, new[] { 0, 1 });
                var testData = cache.AsEnumerable<SentimentData>(env, false);

                if (engine == "mlnet")
                {
                    Console.WriteLine("engine={0} N={1} ncall={2} each={3} cacheScikit={4}", engine, N, ncall, getterEachTime, cacheScikit);
                    var model = env.CreatePredictionEngine<SentimentData, SentimentPrediction>(scorer);
                    var sw = new Stopwatch();
                    for (int call = 1; call <= ncall; ++call)
                    {
                        sw.Reset();
                        sw.Start();
                        for (int i = 0; i < N; ++i)
                            foreach (var input in testData)
                                model.Predict(input);
                        sw.Stop();
                        times.Add(new Tuple<int, TimeSpan, int>(N, sw.Elapsed, call));
                    }
                }
                else if (engine == "scikit")
                {
                    Console.WriteLine("engine={0} N={1} ncall={2} each={3} cacheScikit={4}", engine, N, ncall, getterEachTime, cacheScikit);
                    var model = new ValueMapperPredictionEngine<SentimentData>(env, scorer, getterEachTime: getterEachTime, conc: conc);
                    var output = new ValueMapperPredictionEngine<SentimentData>.PredictionTypeForBinaryClassification();
                    var sw = new Stopwatch();
                    for (int call = 1; call <= ncall; ++call)
                    {
                        if (getterEachTime && call >= 2)
                            break;
                        sw.Reset();
                        sw.Start();
                        for (int i = 0; i < N; ++i)
                            foreach (var input in testData)
                                model.Predict(input, ref output);
                        sw.Stop();
                        times.Add(new Tuple<int, TimeSpan, int>(N, sw.Elapsed, call));
                    }
                }
                else
                    throw new NotImplementedException($"Unknown engine '{engine}'.");
            }
            return times;
        }

        public static DataFrame TestScikitAPI_EngineSimpleTrainAndPredict(string engine, int th, int N, int ncall, bool cacheScikit)
        {
            var dico = new Dictionary<Tuple<int, string, bool, int, int>, double>();
            var scorer = _TrainSentiment();
            foreach (var each in new[] { false })
                foreach (var res in _MeasureTime(th, each, engine, scorer, N, ncall, cacheScikit))
                    dico[new Tuple<int, string, bool, int, int>(res.Item1, engine, each, th, res.Item3)] = res.Item2.TotalSeconds;
            var df = DataFrameIO.Convert(dico, "N", "engine", "getterEachTime", "number of threads", "call", "time(s)");
            return df;
        }
    }
}
