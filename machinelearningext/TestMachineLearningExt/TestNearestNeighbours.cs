// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.TestHelper;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Ext.PipelineHelper;
using Microsoft.ML.Ext.NearestNeighbours;
using Microsoft.ML.Ext.DataManipulation;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestNearestNeighbours
    {
        static void TrainkNNBinaryClassification(int k, NearestNeighborsWeights weight, int threads, float ratio = 0.2f,
                                                 string distance = "L2", int conc = 0)
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris_binary.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment(conc: conc);
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            if (distance == "cosine")
                concat = env.CreateTransform("Scaler{col=Features}", concat);
            var roles = env.CreateExamples(concat, "Features", "Label");
            string modelDef;
            modelDef = string.Format("knn{{k={0} weight={1} nt={2} distance={3} seed=1}}", k,
                                     weight == NearestNeighborsWeights.distance ? "distance" : "uniform", threads, distance);
            var trainer = env.CreateTrainer(modelDef);
            using (var ch = env.Start("test"))
            {
                var pred = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                            PredictionKind.BinaryClassification, true, ratio: ratio);
                ch.Done();
            }
        }

        public static void TrainkNNBinaryClassificationId(int k, NearestNeighborsWeights weight, int threads, float ratio = 0.2f,
                                                        string distance = "L2")
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris_binary_id.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 col=Uid:I8:5 header=+}",
                new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            if (distance == "cosine")
                concat = env.CreateTransform("Scaler{col=Features}", concat);
            var roles = env.CreateExamples(concat, "Features", "Label");
            string modelDef;
            modelDef = string.Format("knn{{k={0} weight={1} nt={2} distance={3} id=Uid}}", k,
                                     weight == NearestNeighborsWeights.distance ? "distance" : "uniform", threads, distance);
            var trainer = env.CreateTrainer(modelDef);
            using (var ch = env.Start("test"))
            {
                var pred = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                        PredictionKind.BinaryClassification, true, ratio: ratio);
                ch.Done();
            }
        }

        public static void TrainkNNMultiClassification(int k, NearestNeighborsWeights weight, int threads, float ratio = 0.2f,
                                                       string distance = "L2")
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            var roles = env.CreateExamples(concat, "Features", "Label");
            string modelDef;
            modelDef = string.Format("knnmc{{k={0} weight={1} nt={2} distance={3}}}", k,
                                     weight == NearestNeighborsWeights.distance ? "distance" : "uniform", threads, distance);
            var trainer = env.CreateTrainer(modelDef);
            using (var ch = env.Start("test"))
            {
                var pred = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                            PredictionKind.MultiClassClassification, true, ratio: ratio);
                ch.Done();
            }
        }

        public static void TrainkNNTransformId(int k, NearestNeighborsWeights weight, int threads, string distance = "L2")
        {
            var methodName = string.Format("{0}-k{1}-W{2}-T{3}-D{4}", System.Reflection.MethodBase.GetCurrentMethod().Name, k, weight, threads, distance);
            var dataFilePath = FileHelper.GetTestFile("iris_binary_id.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = k == 1 ? EnvHelper.NewTestEnvironment(conc: 1) : EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 col=Uid:I8:5 header=+}",
                new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            if (distance == "cosine")
                concat = env.CreateTransform("Scaler{col=Features}", concat);
            concat = env.CreateTransform("knntr{k=5 id=Uid}", concat);
            long nb = DataViewUtils.ComputeRowCount(concat);
            if (nb == 0)
                throw new System.Exception("Empty pipeline.");

            using (var cursor = concat.GetRowCursor(i => true))
            {
                var getdist = cursor.GetGetter<VBuffer<float>>(7);
                var getid = cursor.GetGetter<VBuffer<DvInt8>>(8);
                var ddist = new VBuffer<float>();
                var did = new VBuffer<DvInt8>();
                while (cursor.MoveNext())
                {
                    getdist(ref ddist);
                    getid(ref did);
                    if (!ddist.IsDense || !did.IsDense)
                        throw new System.Exception("not dense");
                    if (ddist.Count != did.Count)
                        throw new System.Exception("not the same dimension");
                    for (int i = 1; i < ddist.Count; ++i)
                    {
                        if (ddist.Values[i - 1] > ddist.Values[i])
                            throw new System.Exception("not sorted");
                        if (did.Values[i].RawValue % 2 != 1)
                            throw new System.Exception("wrong id");
                    }
                }
            }

            TestTransformHelper.SerializationTestTransform(env, outModelFilePath, concat, loader, outData, outData2, false);
        }

        [TestMethod]
        public void TrainkNNBinaryClassification1()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f);
        }

        [TestMethod]
        public void TrainkNNBinaryClassification2()
        {
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 1);
        }

        [TestMethod]
        public void TrainkNNBinaryClassification5()
        {
            TrainkNNBinaryClassification(5, NearestNeighborsWeights.uniform, 1);
        }

        [TestMethod]
        public void TrainkNNBinaryClassificationId()
        {
            TrainkNNBinaryClassificationId(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f);
        }

        [TestMethod]
        public void TrainkNNBinaryClassificationCosine()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f, distance: "cosine", conc: 1);
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 1, distance: "cosine", conc: 1);
        }

        [TestMethod]
        public void TrainkNNBinaryClassificationL1()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f, distance: "L1");
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 1, distance: "L1");
            TrainkNNBinaryClassification(10, NearestNeighborsWeights.uniform, 1, distance: "L1");
        }

        [TestMethod]
        public void TrainkNNBinaryClassificationMultiThread()
        {
            TrainkNNBinaryClassification(1, NearestNeighborsWeights.uniform, 2, ratio: 0.05f);
            TrainkNNBinaryClassification(2, NearestNeighborsWeights.uniform, 2);
            TrainkNNBinaryClassification(10, NearestNeighborsWeights.uniform, 2);
        }

        [TestMethod]
        public void TrainkNNMultiClassification()
        {
            TrainkNNMultiClassification(1, NearestNeighborsWeights.uniform, 1, ratio: 0.05f);
            TrainkNNMultiClassification(2, NearestNeighborsWeights.uniform, 1);
            TrainkNNMultiClassification(10, NearestNeighborsWeights.uniform, 1);
        }

        [TestMethod]
        public void TrainkNNMultiClassificationMultiThread()
        {
            TrainkNNMultiClassification(1, NearestNeighborsWeights.uniform, 2, ratio: 0.05f);
            TrainkNNMultiClassification(2, NearestNeighborsWeights.uniform, 2);
            TrainkNNMultiClassification(10, NearestNeighborsWeights.uniform, 2);
        }

        [TestMethod]
        public void TrainkNNTransformId()
        {
            TrainkNNTransformId(1, NearestNeighborsWeights.uniform, 2);
            TrainkNNTransformId(2, NearestNeighborsWeights.uniform, 2);
            TrainkNNTransformId(10, NearestNeighborsWeights.uniform, 2);
        }

        [TestMethod]
        public void TestNearestNeighborsLPTr()
        {
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Microsoft.ML.Ext.EntryPoints.NearestNeighborsTr("Features"));
            learningPipeline.Add(new StochasticDualCoordinateAscentRegressor());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 18));
        }

        [TestMethod]
        public void TestNearestNeighborsLPBc()
        {
            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var iris = FileHelper.GetTestFile("iris_binary.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Microsoft.ML.Ext.EntryPoints.NearestNeighborsBc());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 9));
        }
    }
}
