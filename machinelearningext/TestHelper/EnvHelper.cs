﻿// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.CommandLine;


namespace Microsoft.ML.Ext.TestHelper
{
    public static class EnvHelper
    {
        public static TlcEnvironment NewTestEnvironment(int? seed = null, bool verbose = false,
                            MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                            int conc = 0, TextWriter outWriter = null, TextWriter errWriter = null)
        {
            if (!seed.HasValue)
                seed = 42;
            if (outWriter == null)
                outWriter = new StreamWriter(new MemoryStream());
            if (errWriter == null)
                errWriter = new StreamWriter(new MemoryStream());
            return new TlcEnvironment(seed, verbose, sensitivity, conc, outWriter, errWriter);
        }

        public static ITrainer CreateTrainer(this IHostEnvironment env, string settings, out string loadName, params object[] extraArgs)
        {
            var sc = SubComponent.Parse<ITrainer, SignatureTrainer>(settings);
            loadName = sc.Kind;
            return sc.CreateInstance(env, extraArgs);
        }
    }
}
