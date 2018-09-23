// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Helper for numeric purpose.
    /// </summary>
    public static class NumericHelper
    {
        public static double AlmostEqual(double exp, double res, double precision = 1e-7)
        {
            if (exp == 0)
                return Math.Abs(res) < precision ? 0f : Math.Abs(res);
            else
            {
                double delta = exp - res;
                double rel = (Math.Abs(delta) / Math.Abs(exp));
                return rel < precision ? 0 : rel;
            }
        }

        public static float AlmostEqual(float exp, float res, float precision = 1e-7f)
        {
            if (exp == 0)
                return Math.Abs(res) < precision ? 0f : Math.Abs(res);
            else
            {
                float delta = exp - res;
                float rel = (Math.Abs(delta) / Math.Abs(exp));
                return rel < precision ? 0f : rel;
            }
        }

        public static double AlmostEqual(IEnumerable<double> exp, IEnumerable<double> res, double precision = 1e-7)
        {
            return Enumerable.Max(Enumerable.Zip(exp, res, (a, b) => AlmostEqual(a, b, precision)), c => c);
        }

        public static float AlmostEqual(IEnumerable<float> exp, IEnumerable<float> res, float precision = 1e-7f)
        {
            return Enumerable.Max(Enumerable.Zip(exp, res, (a, b) => AlmostEqual(a, b, precision)), c => c);
        }

        public static double AssertAlmostEqual(bool[] a1, bool[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if (a1[i].CompareTo(a2[i]) != 0)
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static double AssertAlmostEqual(DvText[] a1, DvText[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if (a1[i].CompareTo(a2[i]) != 0)
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static double AssertAlmostEqual(int[] a1, int[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if (a1[i] != a2[i])
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static double AssertAlmostEqual(long[] a1, long[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if (a1[i] != a2[i])
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static double AssertAlmostEqual(uint[] a1, uint[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if ((double)a1[i] - (double)a2[i] >= precision)
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static double AssertAlmostEqual(float[] a1, float[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if ((double)a1[i] - (double)a2[i] >= precision)
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static double AssertAlmostEqual(double[] a1, double[] a2, double precision = 1e-5, bool exc = true)
        {
            if (a1.Length != a2.Length)
                throw new DataValueError($"Columns have different length {a1.Length} != {a2.Length}.");
            for (int i = 0; i < a1.Length; ++i)
                if ((double)a1[i] - (double)a2[i] >= precision)
                    if (exc)
                        throw new DataValueError($"Values are different at row {i}: {a1[i]} != {a2[i]}.");
                    else
                        return 1;
            return 0;
        }

        public static float[] Convert(int[] data, float missingValue = float.NaN)
        {
            if (data is null)
                return null;
            var res = new float[data.Length];
            for (int i = 0; i < data.Length; ++i)
                res[i] = (float)data[i];
            return res;
        }
    }
}
