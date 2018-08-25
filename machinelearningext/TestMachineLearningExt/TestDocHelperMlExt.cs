// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using DocHelperMlExt;

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
    }
}
