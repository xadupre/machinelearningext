// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.PipelineHelper;
using Scikit.ML.TestHelper;
using Scikit.ML.DataManipulation;
using Scikit.ML.ScikitAPI;
using Legacy = Microsoft.ML.Legacy;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestMLNet
    {
        [TestMethod]
        public void TestTreePathInnerAPI()
        {
            using (var env = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var iris = FileHelper.GetTestFile("iris.txt");
                var df = DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });
                using (var pipe = new ScikitPipeline(new[] { "Concat{col=Feature:Sepal_length,Sepal_width}",
                                                             "TreeFeat{tr=ft{iter=2} lab=Label feat=Feature}"}))
                {
                    pipe.Train(df);
                    var scorer = pipe.Predict(df);
                    var dfout = DataFrameIO.ReadView(scorer);
                    Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 31));
                    var outfile = FileHelper.GetOutputFile("iris_path.txt", methodName);
                    dfout.ToCsv(outfile);
                    Assert.IsTrue(File.Exists(outfile));
                }
            }
        }

        [TestMethod]
        public void TestTreePathNewAPI()
        {
            using (var env = EnvHelper.NewTestEnvironment(conc: 1))
            {
                var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
                var iris = FileHelper.GetTestFile("iris.txt");
                var df = DataFrameIO.ReadCsv(iris, sep: '\t', dtypes: new ColumnType[] { NumberType.R4 });
                var importData = df.EPTextLoader(iris, sep: '\t', header: true);
                var learningPipeline = new GenericLearningPipeline();
                learningPipeline.Add(importData);
                learningPipeline.Add(new Legacy.Transforms.ColumnConcatenator("Features", "Sepal_length", "Sepal_width"));
                learningPipeline.Add(new Legacy.Trainers.StochasticDualCoordinateAscentRegressor() { MaxIterations = 2 });
                var predictor = learningPipeline.Train();
                var predictions = predictor.Predict(df);
                var dfout = DataFrameIO.ReadView(predictions);
                Assert.AreEqual(dfout.Shape, new Tuple<int, int>(150, 8));
                var outfile = FileHelper.GetOutputFile("iris_path.txt", methodName);
                dfout.ToCsv(outfile);
                Assert.IsTrue(File.Exists(outfile));
            }
        }
    }
}
