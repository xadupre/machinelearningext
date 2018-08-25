// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Tools;
using Scikit.ML.DataManipulation;
using Scikit.ML.ScikitAPI;


namespace DocHelperMlExt
{
    /// <summary>
    /// Raised when a script cannot be executed.
    /// </summary>
    public class MamlException : Exception
    {
        public MamlException(string msg) : base(msg)
        {
        }
    }

    public class ExampleVector
    {
        [VectorType(3)]
        public float[] X;
    }

    /// <summary>
    /// Helpers to run scripts through maml.
    /// </summary>
    public static class MamlHelper
    {
        public static void TestScikitAPI()
        {
            var inputs = new[] {
                new ExampleVector() { X = new float[] { 1, 10, 100 } },
                new ExampleVector() { X = new float[] { 2, 3, 5 } },
                new ExampleVector() { X = new float[] { 2, 4, 5 } },
                new ExampleVector() { X = new float[] { 2, 4, 7 } },
            };

            var inputs2 = new[] {
                new ExampleVector() { X = new float[] { -1, -10, -100 } },
                new ExampleVector() { X = new float[] { -2, -3, -5 } },
                new ExampleVector() { X = new float[] { 3, 4, 5 } },
                new ExampleVector() { X = new float[] { 3, 4, 7 } },
            };

            using (var host = new TlcEnvironment(conc: 1))
            {
                var data = host.CreateStreamingDataView(inputs);
                var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host);
                var predictor = pipe.Train(data, feature: "X");
                if (predictor == null)
                    throw new Exception("Test failed: no predictor.");
                var data2 = host.CreateStreamingDataView(inputs2);
                var predictions = pipe.Predict(data2);
                var df = DataFrame.ReadView(predictions);
                if (df.Shape.Item1 != 4 || df.Shape.Item2 != 12)
                    throw new Exception("Test failed: prediction failed.");
                var dfs = df.ToString();
                var dfs2 = dfs.Replace("\n", ";");
                if (!dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"))
                    throw new Exception("Test failed: prediction failed (header).");
            }
        }

        public static string MamlAll(string script, bool catch_output)
        {
            int errCode;
            string res;
            if (catch_output)
            {
                using (var capture = new StdCapture())
                {
                    errCode = Maml.MainAll(script);
                    var sout = capture.StdOut;
                    var serr = capture.StdErr;
                    if (string.IsNullOrEmpty(serr))
                        res = sout;
                    else
                        res = $"--OUT--\n{sout}\n--ERR--\n{serr}";
                }
            }
            else
            {
                errCode = Maml.MainAll(script);
                res = string.Empty;
            }
            if (errCode != 0)
                throw new MamlException($"Unable to run script, error code={errCode}\n{script}\n{res}");
            return res;
        }
    }
}
