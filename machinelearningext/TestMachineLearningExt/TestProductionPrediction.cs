// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineLambdaTransforms;
using Scikit.ML.ProductionPrediction;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestProductionPrediction
    {
        [TestMethod]
        public void TestTransform2ValueMapperMultiThread()
        {
            var env = EnvHelper.NewTestEnvironment();
            var host = env.Register("unittest");

            var inputs = new[] {
                new InputOutput { X = new float[] { 0, 1 }, Y=10 },
                new InputOutput { X = new float[] { 2, 3 }, Y=100 }
            };

            var data = host.CreateStreamingDataView(inputs);

            var trv = LambdaTransform.CreateMap(host, data,
                                        (InputOutput src, InputOutput dst) =>
                                        {
                                            dst.X = new float[] { src.X[0] + (float)src.Y, src.X[1] - (float)src.Y };
                                        });

            var ino = new InputOutput { X = new float[] { -5, -5 }, Y = 3 };
            var inob = new VBuffer<float>(2, ino.X);
            var ans = new VBuffer<float>();

            var cursor = data.GetRowCursor(i => true);
            var gety = cursor.GetGetter<int>(1);

            var valueMapper = new ValueMapperFromTransform<VBuffer<float>>(host, trv, null, "X", "X", cursor);
            var mapper = valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();

            var listy = new List<int>();
            var listx = new List<float>();
            int y = 0;
            while (cursor.MoveNext())
            {
                mapper(ref inob, ref ans);
                gety(ref y);
                if (ans.Count != 2)
                    throw new Exception("Issue with dimension.");
                listx.AddRange(ans.Values);
                listy.Add((int)y);
            }
            if (listy.Count != 2)
                throw new Exception("Issue with dimension.");
            if (listy[0] != 10 || listy[1] != 100)
                throw new Exception("Issue with values.");
            if (listx.Count != 4)
                throw new Exception("Issue with dimension.");
            if (listx[0] != 5)
                throw new Exception("Issue with values.");
            if (listx[1] != -15)
                throw new Exception("Issue with values.");
            if (listx[2] != 95)
                throw new Exception("Issue with values.");
            if (listx[3] != -105)
                throw new Exception("Issue with values.");
            if (inob.Count != 2)
                throw new Exception("Issue with dimension.");
            if (inob.Values[0] != -5)
                throw new Exception("Values were overwritten.");
            if (inob.Values[0] != -5)
                throw new Exception("Values were overwritten.");
        }

        [TestMethod]
        public void TestTransform2ValueMapperSingleThread()
        {
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var host = env.Register("unittest");

            var inputs = new[] {
                new InputOutput { X = new float[] { 0, 1 }, Y=10 },
                new InputOutput { X = new float[] { 2, 3 }, Y=100 }
            };

            var data = host.CreateStreamingDataView(inputs);

            var trv = LambdaTransform.CreateMap(host, data,
                                        (InputOutput src, InputOutput dst) =>
                                        {
                                            dst.X = new float[] { src.X[0] + (float)src.Y, src.X[1] - (float)src.Y };
                                        });

            var inos = new InputOutput[] {new InputOutput { X = new float[] { -5, -5 }, Y = 3 },
                new InputOutput { X = new float[] { -6, -6 }, Y = 30 } };
            var ans = new VBuffer<float>();

            var cursor = data.GetRowCursor(i => true);
            var gety = cursor.GetGetter<int>(1);

            var valueMapper = new ValueMapperFromTransform<VBuffer<float>>(host, trv, null, "X", "X", cursor, true);
            var mapper = valueMapper.GetMapper<VBuffer<float>, VBuffer<float>>();

            var listy = new List<int>();
            var listx = new List<float>();
            int y = 0;
            int tour = 0;
            while (cursor.MoveNext())
            {
                var temp = new VBuffer<float>(2, inos[tour++].X);
                mapper(ref temp, ref ans);
                gety(ref y);
                if (ans.Count != 2)
                    throw new Exception("Issue with dimension.");
                listx.AddRange(ans.Values);
                listy.Add((int)y);
            }
            if (listy.Count != 2)
                throw new Exception("Issue with dimension.");
            if (listy[0] != 10 || listy[1] != 100)
                throw new Exception("Issue with values.");
            if (listx.Count != 4)
                throw new Exception("Issue with dimension.");
            if (listx[0] != 5)
                throw new Exception("Issue with values.");
            if (listx[1] != -15)
                throw new Exception("Issue with values.");
            if (listx[2] != 94)
                throw new Exception("Issue with values.");
            if (listx[3] != -106)
                throw new Exception("Issue with values.");
        }

        public class ValueMapperExample
        {
            public class Input
            {
                [VectorType(9)]
                public float[] Features;
            }

            IHostEnvironment _env;
            IDataView _transforms;
            Predictor _predictor;
            ValueMapper<VBuffer<float>, float> _mapper;

            public ValueMapperExample(string modelName, string features, bool getterEachTime)
            {
                _env = EnvHelper.NewTestEnvironment();
                _predictor = _env.LoadPredictorOrNull(File.OpenRead(modelName));
                var inputs = new Input[0];

                var view = _env.CreateStreamingDataView<Input>(inputs);
                _transforms = ComponentCreation.LoadTransforms(_env, File.OpenRead(modelName), view);
                var data = _env.CreateExamples(_transforms, features);
                var scorer = _env.CreateDefaultScorer(data, _predictor);

                var valueMapper = new ValueMapperFromTransform<VBuffer<float>>(_env,
                                    scorer, view, "Features", "Probability", null, getterEachTime);
                _mapper = valueMapper.GetMapper<VBuffer<float>, float>();
            }

            public float Predict(float[] features)
            {
                float res = 0f;
                var buf = new VBuffer<float>(features.Length, features);
                _mapper(ref buf, ref res);
                return res;
            }
        }

        public class PredictionEngineExample
        {
            IHostEnvironment _env;
            SimplePredictionEngine _predictor;

            public PredictionEngineExample(string modelName)
            {
                _env = EnvHelper.NewTestEnvironment();
                _predictor = _env.CreateSimplePredictionEngine(File.OpenRead(modelName), 9);
            }

            public Tuple<float, float> Predict(float[] features)
            {
                var res = _predictor.Predict(features);
                return new Tuple<float, float>(res.Score, res.Probability);
            }
        }

        [TestMethod]
        public void TestTransform2ValueMapperSingleThreadSimple()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            var example = new ValueMapperExample(name, "Features", true);
            var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
            var ans = example.Predict(feat);
            var ans2 = example.Predict(feat);
            if (ans != ans2)
                throw new Exception(string.Format("Issue {0} != {1}", ans, ans2));
            var engine = new PredictionEngineExample(name);
            var ans6 = engine.Predict(feat);
            if (ans6.Item2 != ans2)
                throw new Exception(string.Format("Issue {0} != {1}", ans, ans6.Item2));

            feat = new float[] { 5, 1, 1, 1, 2, -1, 3, 1, 1 };
            ans = example.Predict(feat);
            ans2 = example.Predict(feat);
            if (ans != ans2)
                throw new Exception(string.Format("Issue {0} != {1}", ans, ans2));
            ans6 = engine.Predict(feat);
            if (ans6.Item2 != ans)
                throw new Exception(string.Format("Issue {0} != {1}", ans, ans6.Item2));

            var exampleNo = new ValueMapperExample(name, "Features", false);
            var ans3 = example.Predict(feat);
            if (ans != ans3)
                throw new Exception(string.Format("Issue {0} != {1}", ans, ans3));
        }

        static void RunValueMapperExample(ValueMapperExample run, int n)
        {
            var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
            for (int i = 0; i < n; ++i)
            {
                feat[0] = i;
                run.Predict(feat);
            }
        }

        static void RunPredictionEngineExample(PredictionEngineExample run, int n)
        {
            var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
            for (int i = 0; i < n; ++i)
            {
                feat[0] = i;
                run.Predict(feat);
            }
        }

        [TestMethod]
        public void TestTransform2ValueMapperMeasuringTime()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            var example = new ValueMapperExample(name, "Features", true);
            var engine = new PredictionEngineExample(name);
            RunValueMapperExample(example, 10000);
            RunPredictionEngineExample(engine, 10000);
        }

        [TestMethod]
        public void TestValueMapperPredictionEngine()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            var engine = new ValueMapperPredictionEngine<ValueMapperExample.Input>(EnvHelper.NewTestEnvironment(), name);
            var feat = new float[] { 5, 1, 1, 1, 2, 1, 3, 1, 1 };
            for (int i = 0; i < 1000; ++i)
            {
                feat[0] = i;
                engine.Predict(feat);
            }
        }

        public class ValueMapperPredictionEngineExample
        {
            public class InputRow
            {
                [VectorType(9)]
                public float[] Features;
            }

            ValueMapperPredictionEngine<InputRow> engine;

            public ValueMapperPredictionEngineExample()
            {
            }

            public void Init(string modelName)
            {
                try
                {
                    var env = EnvHelper.NewTestEnvironment();
                    engine = new ValueMapperPredictionEngine<InputRow>(env,
                                                modelName, "Features", "Probability", false);
                }
                catch (Exception e)
                {
                    throw new Exception("erreur", e);
                }

            }

            public float Predict(float[] features)
            {
                return engine.Predict(features);
            }
        }

        [TestMethod]
        public void TestValueMapperPredictionEngineNotebook()
        {
            var name = FileHelper.GetTestFile("bc-lr.zip");
            var engine = new ValueMapperPredictionEngineExample();
            engine.Init(name);
            var res = engine.Predict(new float[] { 8, 10, 10, 8, 7, 10, 9, 7, 1 });
            if (res < 0)
                throw new Exception("unexpected");
        }



        [TestMethod]
        public void TestLambdaColumnPassThroughTransform()
        {
            var host = EnvHelper.NewTestEnvironment();

            var inputs = new InputOutputU[] {
                new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
            };

            var data = host.CreateStreamingDataView(inputs);
            var lambdaView = LambdaColumnHelper.Create<VBuffer<float>, VBuffer<float>>(host,
                            "Lambda", data, "X", "XX", new VectorType(NumberType.R4, 2),
                            new VectorType(NumberType.R4, 2),
                            (ref VBuffer<float> src, ref VBuffer<float> dst) =>
                            {
                                dst = new VBuffer<float>(2, new float[2]);
                                dst.Values[0] = src.Values[0] + 1f;
                                dst.Values[1] = src.Values[1] + 1f;
                            });

            using (var cursor = lambdaView.GetRowCursor(i => true))
            {
                var labelGetter = cursor.GetGetter<uint>(1);
                var floatGetter = cursor.GetGetter<VBuffer<float>>(2);
                var array = new VBuffer<float>();
                var cont = new List<Tuple<float, float>>();
                while (cursor.MoveNext())
                {
                    uint got = 0;
                    labelGetter(ref got);
                    floatGetter(ref array);
                    cont.Add(new Tuple<float, float>(array.Values[0], array.Values[1]));
                }

                if (cont.Count != 3)
                    throw new Exception("Should be 3");
                for (int i = 0; i < cont.Count; ++i)
                {
                    if (cont[i].Item1 < 1)
                        throw new Exception("Values should be > 1");
                }
            }
        }
    }
}
