// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using Scikit.ML.DocHelperMlExt;
using Scikit.ML.TestHelper;

namespace TestMachineLearningExt
{
    [TestClass]
    public class TestDocHelperMlExt
    {
        [TestMethod]
        public void TestMamlHelperHelp()
        {
            var sout = MamlHelper.MamlScript("?", true);
            if (!sout.Contains("Train"))
                throw new Exception(sout);
        }

        [TestMethod]
        public void TestMamlHelperTest()
        {
            MamlHelper.TestScikitAPI();
        }

        [TestMethod]
        public void TestMamlHelperTest2_AssemblyList()
        {
            MamlHelper.TestScikitAPI2();
            LinkHelper._Immutable();
            var ass = MamlHelper.GetLoadedAssemblies();
            Assert.IsTrue(ass.Length > 0);
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var outData = FileHelper.GetOutputFile("loaded_assemblies.txt", methodName);
            File.WriteAllText(outData, string.Join("\n", ass));
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
