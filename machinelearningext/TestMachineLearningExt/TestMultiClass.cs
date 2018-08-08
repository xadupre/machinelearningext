// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.TestHelper;
using Scikit.ML.MultiClass;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestMultiClass
    {
        #region MultiToBinary transform

        public static void TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm algo, int max)
        {
            var host = EnvHelper.NewTestEnvironment();

            var inputs = new InputOutputU[] {
                new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
            };

            var data = host.CreateStreamingDataView(inputs);

            var args = new MultiToBinaryTransform.Arguments { label = "Y", algo = algo, maxMulti = max };
            var multiplied = new MultiToBinaryTransform(host, args, data);

            using (var cursor = multiplied.GetRowCursor(i => true))
            {
                var labelGetter = cursor.GetGetter<uint>(1);
                var binGetter = cursor.GetGetter<DvBool>(2);
                var cont = new List<Tuple<uint, DvBool>>();
                DvBool bin = DvBool.NA;
                while (cursor.MoveNext())
                {
                    uint got = 0;
                    labelGetter(ref got);
                    binGetter(ref bin);
                    cont.Add(new Tuple<uint, DvBool>(got, bin));
                }

                if (max >= 3)
                {
                    if (cont.Count != 9)
                        throw new Exception("It should be 9.");
                    if (algo == MultiToBinaryTransform.MultiplicationAlgorithm.Default)
                    {
                        for (int i = 0; i < 3; ++i)
                        {
                            var co = cont.Where(c => c.Item1 == (uint)i && c.Item2.IsTrue);
                            if (co.Count() != 1)
                                throw new Exception(string.Format("Unexpected number of true labels for class {0} - algo={1} - max={2}", i, algo, max));
                        }
                    }
                }
                else
                {
                    if (cont.Count != 3 * max)
                        throw new Exception(string.Format("It should be {0}.", 3 * max));
                }
            }
        }

        public static void TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm algo, int max)
        {
            var host = EnvHelper.NewTestEnvironment();

            var inputs = new InputOutputU[] {
                new InputOutputU() { X = new float[] { 0.1f, 1.1f }, Y = 0 },
                new InputOutputU() { X = new float[] { 0.2f, 1.2f }, Y = 1 },
                new InputOutputU() { X = new float[] { 0.3f, 1.3f }, Y = 2 }
            };

            var data = host.CreateStreamingDataView(inputs);

            var args = new MultiToBinaryTransform.Arguments { label = "Y", algo = algo, maxMulti = max };
            var multiplied = new MultiToBinaryTransform(host, args, data);

            using (var cursor = multiplied.GetRowCursor(i => true))
            {
                var labelGetter = cursor.GetGetter<uint>(1);
                var labelVectorGetter = cursor.GetGetter<VBuffer<DvBool>>(1);
                var labelVectorFloatGetter = cursor.GetGetter<VBuffer<float>>(1);
                var binGetter = cursor.GetGetter<DvBool>(2);
                Contracts.CheckValue(binGetter, "Type mismatch.");
                var cont = new List<Tuple<uint, DvBool>>();
                DvBool bin = DvBool.NA;
                uint got = 0;
                var gotv = new VBuffer<DvBool>();
                var gotvf = new VBuffer<float>();
                while (cursor.MoveNext())
                {
                    labelGetter(ref got);
                    labelVectorGetter(ref gotv);
                    labelVectorFloatGetter(ref gotvf);
                    binGetter(ref bin);
                    cont.Add(new Tuple<uint, DvBool>(got, bin));
                    if (gotv.Length != 3) throw new Exception("Bad dimension (Length)");
                    if (gotv.Count != 1) throw new Exception("Bad dimension (Count)");
                    if (!gotv.Values[0].IsTrue) throw new Exception("Bad value (Count)");
                    if (gotv.Indices[0] != got) throw new Exception("Bad index (Count)");
                    var ar = gotv.DenseValues().ToArray();
                    if (ar.Length != 3) throw new Exception("Bad dimension (dense)");

                    if (gotvf.Length != 3) throw new Exception("Bad dimension (Length)f");
                    if (gotvf.Count != 1) throw new Exception("Bad dimension (Count)f");
                    if (gotvf.Values[0] != 1) throw new Exception("Bad value (Count)f");
                    if (gotvf.Indices[0] != got) throw new Exception("Bad index (Count)f");
                    var ar2 = gotv.DenseValues().ToArray();
                    if (ar2.Length != 3) throw new Exception("Bad dimension (dense)f");
                }

                if (max >= 3)
                {
                    if (cont.Count != 9)
                        throw new Exception("It should be 9.");
                    if (algo == MultiToBinaryTransform.MultiplicationAlgorithm.Default)
                    {
                        for (int i = 0; i < 3; ++i)
                        {
                            var co = cont.Where(c => c.Item1 == (uint)i && c.Item2.IsTrue);
                            if (co.Count() != 1)
                                throw new Exception(string.Format("Unexpected number of true labels for class {0} - algo={1} - max={2}", i, algo, max));
                        }
                    }
                }
                else
                {
                    if (cont.Count != 3 * max)
                        throw new Exception(string.Format("It should be {0}.", 3 * max));
                }
            }
        }

        [TestMethod]
        public void TestTransMultiToBin()
        {
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 5);
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 2);
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 5);
            TestMultiToBinaryTransform(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 2);
        }

        [TestMethod]
        public void TestTransMultiToBinVector()
        {
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 5);
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Default, 2);
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 5);
            TestMultiToBinaryTransformVector(MultiToBinaryTransform.MultiplicationAlgorithm.Reweight, 2);
        }

        #endregion

        #region MultiToBinary Predictors

        public static void TrainMultiToBinaryPredictorDense(string modelName, int threads, bool checkError,
                                                            bool singleColumn, bool shift, bool useUint,
                                                            string reclassPredictor = null)
        {
            var methodName = string.Format("{0}-{1}-V{2}-T{3}-S{4}-RP{5}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    modelName, singleColumn ? "C" : "Vec", threads, shift ? "shift" : "std",
                                    string.IsNullOrEmpty(reclassPredictor) ? "0" : reclassPredictor.Replace("{", "")
                                            .Replace("}", "").Replace("+", "").Replace("-", "").Replace(" ", "").Replace("=", ""));
            var dataFilePath = shift
                ? FileHelper.GetTestFile("mc_iris_shift.txt")
                : FileHelper.GetTestFile("mc_iris.txt");
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment(conc: threads == 1 ? 1 : 0);
            string labelType = useUint ? "U4[0-2]" : "R4";
            string loadSettings = string.Format("Text{{col=Label:{0}:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}}", labelType);
            var loader = env.CreateLoader(loadSettings, new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            var roles = env.CreateExamples(concat, "Features", "Label");
            string modelDef = threads <= 0 ? modelName : string.Format("{0}{{t={1}}}", modelName, threads);
            if (!string.IsNullOrEmpty(reclassPredictor))
                reclassPredictor = " rp=" + reclassPredictor;
            string iova = string.Format("iova{{p={0} sc={1}{2}}}", modelDef, singleColumn ? "+" : "-", reclassPredictor);
            var trainer = env.CreateTrainer(iova);
            using (var ch = env.Start("train"))
            {
                var predictor = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                     PredictionKind.MultiClassClassification, checkError, ratio: 0.1f);
                ch.Done();
            }
        }

        public static void TrainMultiToRankerPredictorDense(string modelName, int threads, bool checkError,
                                                            bool singleColumn, bool shift, bool useUint)
        {
            var methodName = string.Format("{0}-{1}-V{2}-T{3}-S{4}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    modelName, singleColumn ? "C" : "Vec", threads, shift ? "shift" : "std");
            var dataFilePath = shift
                ? FileHelper.GetTestFile("mc_iris_shift.txt")
                : FileHelper.GetTestFile("mc_iris.txt");
            var trainFile = FileHelper.GetOutputFile("iris_train.idv", methodName);
            var testFile = FileHelper.GetOutputFile("iris_test.idv", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment(conc: threads == 1 ? 1 : 0);
            string labelType = useUint ? "U4[0-2]" : "R4";
            string loadSettings = string.Format("Text{{col=Label:{0}:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}}", labelType);
            var loader = env.CreateLoader(loadSettings, new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            var roles = env.CreateExamples(concat, "Features", "Label");
            string modelDef = threads <= 0 ? modelName : string.Format("{0}{{t={1}}}", modelName, threads);
            string additionnal = modelName.Contains("xgbrk") ? " u4=+" : "";
            string iova = string.Format("iovark{{p={0} sc={1}{2}}}", modelDef, singleColumn ? "+" : "-", additionnal);
            var trainer = env.CreateTrainer(iova);
            using (var ch = env.Start("train"))
            {
                var predictor = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                     PredictionKind.MultiClassClassification, checkError, ratio: 0.1f);
                ch.Done();
            }
        }

        public static void TrainMultiToBinaryPredictorSparse(bool singleColumn, bool checkError)
        {
            var methodName = string.Format("{0}-{1}-V{2}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    "lr", singleColumn ? "C" : "Vec");
            var trainFile = FileHelper.GetTestFile("Train-28x28_small.txt");
            var testFile = FileHelper.GetTestFile("Test-28x28_small.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text", new MultiFileSource(trainFile));
            var roles = env.CreateExamples(loader, "Features", "Label");
            var iova = string.Format("iova{{p=lr sc={0}}}", singleColumn ? "+" : "-");
            loader = env.CreateLoader("Text", new MultiFileSource(testFile));
            var trainer = env.CreateTrainer(iova);
            using (var ch = env.Start("train"))
            {
                var predictor = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                     PredictionKind.MultiClassClassification, checkError, ratio: 0.1f);
                ch.Done();
            }
        }

        public static void TrainMultiToRankerPredictorSparse(bool singleColumn, bool checkError)
        {
            var methodName = string.Format("{0}-{1}-V{2}", System.Reflection.MethodBase.GetCurrentMethod().Name,
                                    "lr", singleColumn ? "C" : "Vec");
            var trainFile = FileHelper.GetTestFile("Train-28x28_small.txt");
            var testFile = FileHelper.GetTestFile("Test-28x28_small.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text", new MultiFileSource(trainFile));
            var roles = env.CreateExamples(loader, "Features", "Label");
            var iova = string.Format("iovark{{p=ftrank sc={0}}}", singleColumn ? "+" : "-");
            loader = env.CreateLoader("Text", new MultiFileSource(testFile));
            var trainer = env.CreateTrainer(iova);
            using (var ch = env.Start("train"))
            {
                var predictor = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, predictor, roles, outData, outData2,
                                                     PredictionKind.MultiClassClassification, checkError, ratio: 0.1f);
                ch.Done();
            }
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseLR_SingleColumn()
        {
            TrainMultiToBinaryPredictorDense("lr", -1, false, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseLR_ReclassPredictor()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, false, false, "mlr");
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFTRK_SingleColumn()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseLR_Vector()
        {
            TrainMultiToBinaryPredictorDense("lr", -1, false, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFTRK_Vector()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, false, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_SingleColumn()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumn()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, true, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumn_XGBoost()
        {
            TrainMultiToRankerPredictorDense("xgbrk", -1, true, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_SingleColumnUint()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, false, true);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumnUint()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, true, true, false, true);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_Vector()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_Vector()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, true, false, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_VectorUint()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, false, false, true);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_VectorUint()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, true, false, false, true);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_SingleColumn_Shifted()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, true, true, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_SingleColumn_Shifted()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, true, true, true, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T1_Vector_Shifted()
        {
            TrainMultiToBinaryPredictorDense("ft", 1, true, false, true, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorDenseFT_T1_Vector_Shifted()
        {
            TrainMultiToRankerPredictorDense("ftrank", 1, true, false, true, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorDenseFT_T10()
        {
            TrainMultiToBinaryPredictorDense("ft", 10, true, true, false, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorSparseSingleColumn()
        {
            TrainMultiToBinaryPredictorSparse(true, false);
        }

        [TestMethod]
        public void TrainMultiToRankerPredictorSparseSingleColumn()
        {
            TrainMultiToRankerPredictorSparse(true, false);
        }

        [TestMethod]
        public void TrainMultiToBinaryPredictorSparseVector()
        {
            TrainMultiToBinaryPredictorSparse(false, false);
        }

        #endregion
    }
}
