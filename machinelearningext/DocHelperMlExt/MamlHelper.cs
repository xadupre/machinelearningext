// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.DataManipulation;
using Scikit.ML.ScikitAPI;


namespace Scikit.ML.DocHelperMlExt
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
        #region command line

        /// <summary>
        /// Runs a script. Can change the level of desired information.
        /// </summary>
        /// <param name="script"></param>
        /// <param name="catch_output"></param>
        /// <param name="verbose">2 is default</param>
        /// <returns></returns>
        public static string MamlScriptConsole(string script, bool catch_output, int verbose = 2)
        {
            ILogWriter logout = new LogWriter((string s) =>
            {
                if (s.Contains("Elapsed"))
                    throw new Exception(s);
                Console.Write(s);
            });
            ILogWriter logerr = new LogWriter((string s) =>
            {
                if (s.Contains("Elapsed"))
                    throw new Exception(s);
                Console.Error.Write(s);
            });
            using (var env = new DelegateEnvironment(verbose: 2, outWriter: logout, errWriter: logerr))
                return MamlScript(script, catch_output, env);
        }

        /// <summary>
        /// Runs a command line with ML.Net.
        /// </summary>
        /// <param name="script">script to run</param>
        /// <param name="catch_output">capture output</param>
        /// <param name="env">delegate environment</param>
        /// <returns>output and error if captured</returns>
        public static string MamlScript(string script, bool catch_output, DelegateEnvironment env = null)
        {
            int errCode;
            string res;
            if (catch_output)
            {
                using (var capture = new StdCapture())
                {
                    errCode = DocumentationEnvironmentHelper.MainWithProgress(script, env);
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
                errCode = DocumentationEnvironmentHelper.MainWithProgress(script, env);
                res = string.Empty;
            }
            if (errCode != 0)
                throw new MamlException($"Unable to run script, error code={errCode}\n{script}\n{res}");
            return res;
        }

        #endregion

        #region tests

        public static void TestScikitAPITrain(string name = null)
        {
            var script = "train data=__NAME__ loader=text{col=Label:R4:0 col=Features:R4:1-4 header=+} tr=mlr{maxiter=5} out=logistic_regression.zip";
            name = name ?? Path.Combine("..", "..", "..", "..", "..", "docs", "source", "iris.txt");
            name = Path.GetFullPath(name);
            if (!File.Exists(name))
            {
                TestScikitAPI2();
                return;
            }
            script = script.Replace("__NAME__", name);
            var res = MamlScript(script, true);
            if (string.IsNullOrEmpty(res))
                throw new Exception("Empty output.");
        }

        /// <summary>
        /// Runs a simple test.
        /// </summary>
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

            using (var host = new ConsoleEnvironment(conc: 1))
            {
                ComponentHelper.AddStandardComponents(host);
                var data = host.CreateStreamingDataView(inputs);
                using (var pipe = new ScikitPipeline(new[] { "poly{col=X}" }, "km{k=2}", host))
                {
                    var predictor = pipe.Train(data, feature: "X");
                    if (predictor == null)
                        throw new Exception("Test failed: no predictor.");
                    var data2 = host.CreateStreamingDataView(inputs2);
                    var predictions = pipe.Predict(data2);
                    var df = DataFrameIO.ReadView(predictions);
                    if (df.Shape.Item1 != 4 || df.Shape.Item2 != 12)
                        throw new Exception("Test failed: prediction failed.");
                    var dfs = df.ToString();
                    var dfs2 = dfs.Replace("\n", ";");
                    if (!dfs2.StartsWith("X.0,X.1,X.2,X.3,X.4,X.5,X.6,X.7,X.8,PredictedLabel,Score.0,Score.1;-1,-10,-100,1,10,100,100,1000,10000"))
                        throw new Exception("Test failed: prediction failed (header).");
                }
            }
        }

        /// <summary>
        /// Runs a simple test.
        /// </summary>
        public static void TestScikitAPI2()
        {
            var res = MamlScript("? ap", true);
            if (string.IsNullOrEmpty(res))
                throw new Exception("Empty output.");
        }

        #endregion

        #region inspect

        private static string InfoAssembly(Assembly ass)
        {
            string name = ass.FullName;
            try
            {
                return name + " - " + ass.Location;
            }
            catch (Exception e)
            {
                return name + " - " + e.ToString();
            }
        }

        public static string[] GetLoadedAssemblies()
        {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies().Select(x => InfoAssembly(x)).OrderBy(c => c);
            return assemblies.ToArray();
        }

        /// <summary>
        /// Returns all kinds of models.
        /// </summary>
        public static string[] GetAllKinds()
        {
            using (var env = new ConsoleEnvironment())
            {
                ComponentHelper.AddStandardComponents(env);
                var sigs = env.ComponentCatalog.GetAllSignatureTypes();
                var typeSig = sigs.Select(t => ComponentCatalog.SignatureToString(t).ToLowerInvariant());
                return new HashSet<string>(typeSig.OrderBy(c => c)).ToArray();
            }
        }

        public class ComponentDescription
        {
            public class Argument
            {
                public string Name;
                public string Help;
                public string ShortName;
                public string DefaultValue;
                public CmdParser.ArgInfo.Arg Arg;
            }

            public string Name;
            public object Args;
            public string Description;
            public string[] Aliases;
            public ComponentCatalog.LoadableClassInfo Info;
            public Assembly Assembly;
            public string AssemblyName;
            public Argument[] Arguments;

            public DataFrame ArgsAsDataFrame
            {
                get
                {
                    var df = new DataFrame();
                    df.AddColumn("Name", Arguments.Select(c => c.Name).ToArray());
                    df.AddColumn("ShortName", Arguments.Select(c => c.ShortName).ToArray());
                    df.AddColumn("DefaultValue", Arguments.Select(c => c.DefaultValue).ToArray());
                    df.AddColumn("Help", Arguments.Select(c => c.Help).ToArray());
                    return df;
                }
            }
        }

        /// <summary>
        /// Returns all kinds of models.
        /// </summary>
        public static IEnumerable<ComponentDescription> EnumerateComponents(string kind)
        {
            var kinds = GetAllKinds();
            if (!kinds.Where(c => c == kind).Any())
                throw new ArgumentException($"Unable to find kind '{kind}' in\n{string.Join("\n", kinds)}.");

            using (var env = new ConsoleEnvironment())
            {
                ComponentHelper.AddStandardComponents(env);
                var sigs = env.ComponentCatalog.GetAllSignatureTypes();
                var typeSig = sigs.FirstOrDefault(t => ComponentCatalog.SignatureToString(t).ToLowerInvariant() == kind);
                var typeRes = typeof(object);
                var infos = env.ComponentCatalog.GetAllDerivedClasses(typeRes, typeSig)
                    .Where(x => !x.IsHidden)
                    .OrderBy(x => x.LoadNames[0].ToLowerInvariant());
                foreach (var info in infos)
                {
                    var args = info.CreateArguments();
                    if (args == null)
                    {
                        var cmp = new ComponentDescription()
                        {
                            Args = args,
                            Name = info.UserName,
                            Description = info.Summary,
                            Info = info,
                            Aliases = info.LoadNames.ToArray(),
                            Assembly = null,
                            AssemblyName = "?",
                            Arguments = null,
                        };
                        yield return cmp;
                    }
                    else
                    {
                        var asse = args.GetType().Assembly;

                        var parsedArgs = CmdParser.GetArgInfo(args.GetType(), args).Args;
                        var arguments = new List<ComponentDescription.Argument>();
                        foreach (var arg in parsedArgs)
                        {
                            var a = new ComponentDescription.Argument()
                            {
                                Name = arg.LongName,
                                ShortName = arg.ShortNames == null || !arg.ShortNames.Any() ? null : arg.ShortNames.First(),
                                DefaultValue = arg.DefaultValue == null ? null : arg.DefaultValue.ToString(),
                                Help = arg.HelpText,
                                Arg = arg,
                            };
                            arguments.Add(a);
                        }

                        var cmp = new ComponentDescription()
                        {
                            Args = args,
                            Name = info.UserName,
                            Description = info.Summary,
                            Info = info,
                            Aliases = info.LoadNames.ToArray(),
                            Assembly = asse,
                            AssemblyName = asse.ManifestModule.Name,
                            Arguments = arguments.OrderBy(c => c.Name).ToArray(),
                        };
                        yield return cmp;
                    }
                }
            }
        }

        #endregion
    }
}
