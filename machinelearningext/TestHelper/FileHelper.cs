// See the LICENSE file in the project root for more information.


using System;
using System.IO;
using System.Reflection;
using Microsoft.ML.Runtime.Tools;


namespace Microsoft.ML.Ext.TestHelper
{
    public static class FileHelper
    {
        static bool IsRoot(string root)
        {
            if (!Directory.Exists(root))
                return false;
            var dotRoot = Path.Combine(root, "LICENSE");
            if (!File.Exists(dotRoot))
                return false;
            return true;
        }

        public static string GetRoot()
        {
            string currentDirectory = Directory.GetCurrentDirectory();
            currentDirectory = Path.GetFullPath(currentDirectory);
            var root = currentDirectory;
            while (!string.IsNullOrEmpty(root))
            {
                if (IsRoot(root))
                    return root;
                root = Path.GetDirectoryName(root);
            }
            throw new DirectoryNotFoundException(string.Format("Unable to find root folder from '{0}'", currentDirectory));
        }

        /// <summary>
        /// Returns the TLC version used to build TLC-CONTRIB.
        /// </summary>
        public static string GetUsedTlcVersion()
        {
            return typeof(VersionCommand).GetTypeInfo().Assembly.GetName().Version.ToString();
        }

        /// <summary>
        /// Returns a data relative to folder data.
        /// </summary>
        /// <param name="name">name of the dataset</param>
        /// <returns>full path of the dataset</returns>
        public static string GetTestFile(string name)
        {
            var root = GetRoot();
            var version = GetUsedTlcVersion();
            var build = GetTlcBuild(version);
            // we should be in build/tlccontrib
            var full = name.StartsWith("samples\\")
                            ? Path.Combine(root, build, "tlc", "Samples", "Data", name.Substring("samples\\".Length))
                            : Path.Combine(root, "data", name);
            if (!File.Exists(full))
                throw new FileNotFoundException(string.Format("Unable to find '{0}'\nFull='{1}'\nroot='{2}'\ncurrent='{3}'.",
                                    name, full, root, Path.GetFullPath(Directory.GetCurrentDirectory())));
            return full;
        }

        /// <summary>
        /// Retrieve the build folder, checks its existence.
        /// </summary>
        /// <param name="version"></param>
        public static string GetTlcBuild(string version)
        {
            if (string.IsNullOrEmpty(version))
                throw new Exception("version is null");
            string build = null;
            if (version.Contains(","))
            {
                foreach(var vers in version.Split(','))
                {
                    var maml = Path.Combine(GetRoot(), "build" + vers, "tlc", "maml.exe");
                    if (File.Exists(maml))
                    {
                        build = Path.Combine(GetRoot(), "build" + vers);
                        break;
                    }
                }
            }
            else
                build = Path.Combine(GetRoot(), "build" + version);
            if (!Directory.Exists(build))
                throw new DirectoryNotFoundException(build);
            return build;
        }

        /// <summary>
        /// Creates a folder where the results of the unittests should be placed.
        /// </summary>
        /// <param name="name">name of the output</param>
        /// <param name="testFunction">name of the test</param>
        /// <param name="extended">addition to make to the file</param>
        /// <returns></returns>
        public static string GetOutputFile(string name, string testFunction, params string[] extended)
        {
#if (DEBUG)
            var version = "Debug";
#else
            var version = "Release";
#endif
            var vers = GetUsedTlcVersion();
            var root = GetRoot();
            var tests = Path.Combine(root, "_tests");
            var unittest = Path.Combine(tests, vers, version, testFunction);

            if (!Directory.Exists(unittest))
                Directory.CreateDirectory(unittest);

            if (extended != null && extended.Length > 0)
            {
                var jpl = string.Join("-", extended);
                unittest = Path.Combine(unittest, jpl);
                if (!Directory.Exists(unittest))
                    Directory.CreateDirectory(unittest);
            }
            var full = Path.Combine(unittest, name);
            if (File.Exists(full))
                File.Delete(full);
            return full;
        }
    }
}
