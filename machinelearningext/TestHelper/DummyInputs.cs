// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;


namespace MlFood.ML.TestHelper
{
    public class InputOutput
    {
        [VectorType(2)]
        public float[] X;
        public DvInt4 Y;

        public static InputOutput[] CreateInputs()
        {
            var inputs = new InputOutput[] {
                new InputOutput() { X = new float[] { 0, 1 }, Y = 1 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 0 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 2 },
                new InputOutput() { X = new float[] { 0, 1 }, Y = 3 },
            };
            return inputs;
        }
    }
}
