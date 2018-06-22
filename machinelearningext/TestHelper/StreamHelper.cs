// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;


namespace Microsoft.ML.Ext.TestHelper
{
    public static class StreamHelper
    {
        /// <summary>
        /// Computes the prediction given a model as a zip file
        /// and some data in a view.
        /// </summary>
        public static void SavePredictions(TlcEnvironment env, string modelPath,
                                           string outFilePath, IDataView data)
        {
            using (var fs = File.OpenRead(modelPath))
            {
                var deserializedData = env.LoadTransforms(fs, data);
                var saver2 = env.CreateSaver("Text");
                var columns = new int[deserializedData.Schema.ColumnCount];
                for (int i = 0; i < columns.Length; ++i)
                    columns[i] = i;
                using (var fs2 = File.Create(outFilePath))
                    saver2.SaveData(fs2, deserializedData, columns);
            }
        }

        /// <summary>
        /// Computes the prediction given a model as a zip file
        /// and some data in a view.
        /// </summary>
        public static void SavePredictions(TlcEnvironment env, IDataTransform tr, string outFilePath)
        {
            var saver2 = env.CreateSaver("Text");
            var columns = new int[tr.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = i;
            using (var fs2 = File.Create(outFilePath))
                saver2.SaveData(fs2, tr, columns);
        }

        /// <summary>
        /// Saves a model in a zip file.
        /// </summary>
        public static void SaveModel(TlcEnvironment env, IDataTransform tr, string outModelFilePath)
        {
            using (var ch = env.Start("SaveModel"))
            using (var fs = File.Create(outModelFilePath))
            {
                var trainingExamples = env.CreateExamples(tr, null);
                TrainUtils.SaveModel(env, ch, fs, null, trainingExamples);
            }
        }
    }
}
