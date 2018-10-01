// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using Scikit.ML.DocHelperMlExt;

namespace TestMachineLearningExt
{
    [TestClass]
    public class TestDocHelperMlExt
    {
        [TestMethod]
        public void TestMamlHelperHelp()
        {
            var sout = MamlHelper.MamlAll("?", true);
            if (!sout.Contains("Train"))
                throw new Exception(sout);
        }

        [TestMethod]
        public void TestMamlHelperTest()
        {
            MamlHelper.TestScikitAPI();
        }

        [TestMethod]
        public void TestMamlHelperKinds()
        {
            var kinds = MamlHelper.GetAllKinds();
            Assert.IsTrue(kinds.Length > 0);
            Assert.IsTrue(kinds.Where(c => c == "trainer").Any());
            Assert.IsTrue(kinds.Where(c => c == "datatransform").Any());
        }

        [TestMethod]
        public void TestMamlHelperTrainer()
        {
            var trainers = MamlHelper.EnumerateComponents("trainer").ToArray();
            Assert.IsTrue(trainers.Length > 0);
            var df = trainers.First().ArgsAsDataFrame;
            Assert.AreEqual(df.Shape.Item2, 4);
        }
    }
}
