﻿// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.TestHelper
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

    public class InputOutputU
    {
        [VectorType(2)]
        public float[] X;
        public uint Y;
    }

    public class ExampleA
    {
        [VectorType(2)]
        public float[] X;
    }

    public class ExampleASparse
    {
        [VectorType()]
        public VBuffer<float> X;
    }

    public class ExampleValueMapper : IValueMapper
    {
        public ColumnType InputType { get { return new VectorType(NumberType.R4, 2); } }
        public ColumnType OutputType { get { return NumberType.R4; } }
        public ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>()
        {
            return GetMapper_() as ValueMapper<TSrc, TDst>;
        }

        ValueMapper<VBuffer<float>, float> GetMapper_()
        {
            return (ref VBuffer<float> X, ref float y) => { y = X.Values.Sum(); };
        }
    }
}
