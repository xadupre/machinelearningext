// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Scikit.ML.DataManipulation;
using Scikit.ML.PipelineHelper;
using Scikit.ML.PipelineTransforms;
using Scikit.ML.TestHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestPipelineTransforms
    {
        #region DescribeTransform

        [TestMethod]
        public void TestI_DescribeTransformCode()
        {
            var env = EnvHelper.NewTestEnvironment();
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
        public void TestI_DescribeTransformSaveDataAndZip()
        {
            var env = EnvHelper.NewTestEnvironment();
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

        #endregion

        #region PassThroughTransform

        [TestMethod]
        public void TestI_PassThroughTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
            var tempFile = FileHelper.GetOutputFile("dump.idv", methodName);

            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var loader = env.CreateLoader("Text{col=Label:R4:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 col=Uid:TX:5 header=+}",
                new MultiFileSource(dataFilePath));

            var xf1 = env.CreateTransform("Concat{col=Feat:Slength,Swidth}", loader);
            var xf2 = env.CreateTransform("Scaler{col=Feat}", xf1);
            var xf3 = env.CreateTransform(string.Format("DumpView{{s=+ f={0}}}", tempFile), xf2);
            TestTransformHelper.SerializationTestTransform(env, outModelFilePath, xf3, loader, outData, outData2, false);
            if (!File.Exists(tempFile))
                throw new FileNotFoundException(tempFile);
        }

        [TestMethod]
        public void TestEP_PassThroughTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var iris = FileHelper.GetTestFile("iris.txt");
            var outPass = FileHelper.GetOutputFile("data.idv", methodName);
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.Scaler("Features"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.PassThrough() { Filename = outPass, SaveOnDisk = true });
            learningPipeline.Add(new StochasticDualCoordinateAscentRegressor());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(new Tuple<int, int>(150, 8), dfout.Shape);
            Assert.IsTrue(File.Exists(outPass));
        }

        #endregion

        #region ULabel2Float

        [TestMethod]
        public void TestI_ULabelToR4LabelTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("iris_binary.txt");
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);

            var env = EnvHelper.NewTestEnvironment(conc: 1);
            var loader = env.CreateLoader("Text{col=LabelText:TX:0 col=Slength:R4:1 col=Swidth:R4:2 col=Plength:R4:3 col=Pwidth:R4:4 header=+}",
                new MultiFileSource(dataFilePath));

            var concat = env.CreateTransform("Concat{col=Features:Slength,Swidth}", loader);
            var labelTx = env.CreateTransform("TermTransform{col=LabelU4:LabelText}", concat);
            var labelR4 = env.CreateTransform("U2R4{col=Label:LabelU4}", labelTx);
            var roles = env.CreateExamples(labelR4, "Features", "Label");
            var trainer = env.CreateTrainer("lr");
            using (var ch = env.Start("test"))
            {
                var pred = trainer.Train(env, ch, roles);
                TestTrainerHelper.FinalizeSerializationTest(env, outModelFilePath, pred, roles, outData, outData2,
                                                            trainer.Trainer.PredictionKind, true, ratio: 0.8f);
                ch.Done();
            }
        }

        [TestMethod]
        public void TestEP_ULabelToR4LabelTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var iris = FileHelper.GetTestFile("iris.txt");
            var df = DataFrame.ReadCsv(iris, sep: '\t', dtypes: new DataKind?[] { DataKind.R4 });

            var importData = df.EPTextLoader(iris, sep: '\t', header: true);
            var learningPipeline = new GenericLearningPipeline(conc: 1);
            learningPipeline.Add(importData);
            learningPipeline.Add(new ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.Scaler("Features"));
            learningPipeline.Add(new Scikit.ML.EntryPoints.ULabelToR4Label("Label"));
            learningPipeline.Add(new StochasticDualCoordinateAscentRegressor());
            var predictor = learningPipeline.Train();
            var predictions = predictor.Predict(df);
            var dfout = DataFrame.ReadView(predictions);
            Assert.AreEqual(new Tuple<int, int>(150, 8), dfout.Shape);
        }

        #endregion
    }
}
