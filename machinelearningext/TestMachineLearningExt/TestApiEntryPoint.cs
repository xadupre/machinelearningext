// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using Microsoft.ML.Runtime.Tools;
using Scikit.ML.TestHelper;

namespace TestMachineLearningExt
{
    [TestClass]
    public class TestApiEntryPoint
    {
        [TestMethod]
        public void TestCSGeneratorHelp()
        {
            var cmd = "? CSGenerator";
            using (var std = new StdCapture())
            {
                Maml.MainAll(cmd);
                Assert.IsTrue(std.StdOut.Length > 0);
            }
        }

        [TestMethod]
        public void TestCSGenerator()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var basePath = FileHelper.GetOutputFile("CSharpApiExt.cs", methodName);
            var cmd = $"? generator=cs{{csFilename={basePath} exclude=System.CodeDom.dll}}";
            using (var std = new StdCapture())
            {
                Maml.Main(new[] { cmd });
                Assert.IsTrue(std.StdOut.Length > 0);
                Assert.IsTrue(std.StdErr.Length == 0);
                Assert.IsFalse(std.StdOut.ToLower().Contains("usage"));
            }
            var text = File.ReadAllText(basePath);
            Assert.IsTrue(text.ToLower().Contains("nearest"));
        }
    }
}
