﻿// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.TestHelper
{
    public static class TestTransformHelper
    {
        /// <summary>
        /// Finalize the test on a transform, calls the transform,
        /// saves the data, saves the models, loads it back, saves the data again,
        /// checks the output is the same.
        /// </summary>
        /// <param name="env"></param>
        /// <param name="outModelFilePath"></param>
        /// <param name="transform"></param>
        /// <param name="source">source (view before applying the transform</param>
        /// <param name="outData"></param>
        /// <param name="outData2"></param>
        /// <param name="startsWith">Check that outputs is the same on disk after outputting the transformed data after the model was serialized</param>
        public static void SerializationTestTransform(TlcEnvironment env,
                            string outModelFilePath, IDataTransform transform,
                            IDataView source, string outData, string outData2,
                            bool startsWith = false, bool skipDoubleQuote = false,
                            bool forceDense = false)
        {
            // Saves model.
            var roles = env.CreateExamples(transform, null);
            using (var ch = env.Start("SaveModel"))
            using (var fs = File.Create(outModelFilePath))
                TrainUtils.SaveModel(env, ch, fs, null, roles);
            if (!File.Exists(outModelFilePath))
                throw new FileNotFoundException(outModelFilePath);

            // We load it again.
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var tr2 = env.LoadTransforms(fs, source);
                if (tr2 == null)
                    throw new Exception(string.Format("Unable to load '{0}'", outModelFilePath));
                if (transform.GetType() != tr2.GetType())
                    throw new Exception(string.Format("Type mismatch {0} != {1}", transform.GetType(), tr2.GetType()));
            }

            // Checks the outputs.
            var saver = env.CreateSaver(forceDense ? "Text{dense=+}" : "Text");
            var columns = new int[transform.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = i;
            using (var fs2 = File.Create(outData))
                saver.SaveData(fs2, transform, columns);

            if (!File.Exists(outModelFilePath))
                throw new FileNotFoundException(outData);

            // Check we have the same output.
            using (var fs = File.OpenRead(outModelFilePath))
            {
                var tr = env.LoadTransforms(fs, source);
                saver = env.CreateSaver(forceDense ? "Text{dense=+}" : "Text");
                using (var fs2 = File.Create(outData2))
                    saver.SaveData(fs2, tr, columns);
            }

            var t1 = File.ReadAllLines(outData);
            var t2 = File.ReadAllLines(outData2);
            if (t1.Length != t2.Length)
                throw new Exception(string.Format("Not the same number of lines: {0} != {1}", t1.Length, t2.Length));
            for (int i = 0; i < t1.Length; ++i)
            {
                if (skipDoubleQuote && (t1[i].Contains("\"\"\t\"\"") || t2[i].Contains("\"\"\t\"\"")))
                    continue;
                if ((startsWith && !t1[i].StartsWith(t2[i])) || (!startsWith && t1[i] != t2[i]))
                {
                    if (t1[i].EndsWith("\t5\t0:\"\""))
                    {
                        var a = t1[i].Substring(0, t1[i].Length - "\t5\t0:\"\"".Length);
                        a += "\t\"\"\t\"\"\t\"\"\t\"\"\t\"\"";
                        var b = t2[i];
                        if ((startsWith && !a.StartsWith(b)) || (!startsWith && a != b))
                            throw new Exception(string.Format("Mismatch on line (2) {0}/{3}:\n{1}\n{2}", i, t1[i], t2[i], t1.Length));
                    }
                    else
                        // The test might fail because one side is dense and the other is sparse.
                        throw new Exception(string.Format("Mismatch on line (3) {0}/{3}:\n{1}\n{2}", i, t1[i], t2[i], t1.Length));
                }
            }
        }
    }
}