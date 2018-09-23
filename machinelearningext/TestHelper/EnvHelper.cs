// See the LICENSE file in the project root for more information.

using System.IO;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.TestHelper
{
    public static class EnvHelper
    {
        /// <summary>
        /// Creates a new environment. It should be done
        /// with <tt>using</tt>.
        /// </summary>
        public static ConsoleEnvironment NewTestEnvironment(int? seed = null, bool verbose = false,
                            MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                            int conc = 0, TextWriter outWriter = null, TextWriter errWriter = null)
        {
            if (!seed.HasValue)
                seed = 42;
            if (outWriter == null)
                outWriter = new StreamWriter(new MemoryStream());
            if (errWriter == null)
                errWriter = new StreamWriter(new MemoryStream());
            return new ConsoleEnvironment(seed, verbose, sensitivity, conc, outWriter, errWriter);
        }

        public static ConsoleEnvironment NewTestEnvironment(out StringWriter sout, out StringWriter serr,
                                                        int? seed = null, bool verbose = false,
                                                        MessageSensitivity sensitivity = (MessageSensitivity)(-1),
                                                        int conc = 0)
        {
            var sb = new StringBuilder();
            sout = new StringWriter(sb);
            sb = new StringBuilder();
            serr = new StringWriter(sb);
            return NewTestEnvironment(seed: seed, verbose: verbose, sensitivity: sensitivity,
                                      conc: conc, outWriter: sout, errWriter: serr);
        }
    }
}
