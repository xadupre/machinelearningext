// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using DvText = Scikit.ML.PipelineHelper.DvText;


namespace Scikit.ML.DataManipulation
{
    /// <summary>
    /// Implements aggregated functions for various types.
    /// </summary>
    public static class DataFrameAggFunctions
    {
        public static Func<bool[], bool> GetAggFunction(AggregatedFunction func, bool defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Mean:
                case AggregatedFunction.Count:
                    return (bool[] arr) => { return false; };
                case AggregatedFunction.Max:
                case AggregatedFunction.Sum:
                    return (bool[] arr) => { return arr.Aggregate((a, b) => a | b); };
                case AggregatedFunction.Min:
                    return (bool[] arr) => { return arr.Aggregate((a, b) => a & b); };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }

        public static Func<int[], int> GetAggFunction(AggregatedFunction func, int defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Count:
                    return (int[] arr) => { return arr.Length; };
                case AggregatedFunction.Sum:
                    return (int[] arr) => { return arr.Aggregate((a, b) => a + b); };
                case AggregatedFunction.Min:
                    return (int[] arr) => { return arr.Aggregate((a, b) => Math.Min(a, b)); };
                case AggregatedFunction.Max:
                    return (int[] arr) => { return arr.Aggregate((a, b) => Math.Max(a, b)); };
                case AggregatedFunction.Mean:
                    return (int[] arr) => { return arr.Aggregate((a, b) => a + b) / arr.Length; };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }

        public static Func<uint[], uint> GetAggFunction(AggregatedFunction func, uint defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Count:
                    return (uint[] arr) => { return (uint)arr.Length; };
                case AggregatedFunction.Sum:
                    return (uint[] arr) => { return arr.Aggregate((a, b) => a + b); };
                case AggregatedFunction.Min:
                    return (uint[] arr) => { return arr.Min(); };
                case AggregatedFunction.Max:
                    return (uint[] arr) => { return arr.Max(); };
                case AggregatedFunction.Mean:
                    return (uint[] arr) => { return arr.Aggregate((a, b) => a + b) / (uint)arr.Length; };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }

        public static Func<Int64[], Int64> GetAggFunction(AggregatedFunction func, Int64 defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Count:
                    return (Int64[] arr) => { return (Int64)arr.Length; };
                case AggregatedFunction.Sum:
                    return (Int64[] arr) => { return arr.Aggregate((a, b) => a + b); };
                case AggregatedFunction.Min:
                    return (Int64[] arr) => { return arr.Aggregate((a, b) => Math.Min(a, b)); };
                case AggregatedFunction.Max:
                    return (Int64[] arr) => { return arr.Aggregate((a, b) => Math.Max(a, b)); };
                case AggregatedFunction.Mean:
                    return (Int64[] arr) => { return arr.Aggregate((a, b) => a + b) / arr.Length; };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }

        public static Func<float[], float> GetAggFunction(AggregatedFunction func, float defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Count:
                    return (float[] arr) => { return (float)arr.Length; };
                case AggregatedFunction.Sum:
                    return (float[] arr) => { return arr.Sum(); };
                case AggregatedFunction.Min:
                    return (float[] arr) => { return arr.Min(); };
                case AggregatedFunction.Max:
                    return (float[] arr) => { return arr.Max(); };
                case AggregatedFunction.Mean:
                    return (float[] arr) => { return arr.Sum() / (uint)arr.Length; };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }

        public static Func<double[], double> GetAggFunction(AggregatedFunction func, double defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Count:
                    return (double[] arr) => { return (double)arr.Length; };
                case AggregatedFunction.Sum:
                    return (double[] arr) => { return arr.Sum(); };
                case AggregatedFunction.Min:
                    return (double[] arr) => { return arr.Min(); };
                case AggregatedFunction.Max:
                    return (double[] arr) => { return arr.Max(); };
                case AggregatedFunction.Mean:
                    return (double[] arr) => { return arr.Sum() / (uint)arr.Length; };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }

        public static Func<DvText[], DvText> GetAggFunction(AggregatedFunction func, DvText defaultValue)
        {
            switch (func)
            {
                case AggregatedFunction.Mean:
                case AggregatedFunction.Count:
                    return (DvText[] arr) => { return DvText.NA; };
                case AggregatedFunction.Sum:
                    return (DvText[] arr) => { return arr.Aggregate((a, b) => new DvText(a.ToString() + b.ToString())); };
                case AggregatedFunction.Min:
                    return (DvText[] arr) => { return arr.Aggregate((a, b) => a.CompareTo(b) <= 0 ? a : b); };
                case AggregatedFunction.Max:
                    return (DvText[] arr) => { return arr.Aggregate((a, b) => a.CompareTo(b) >= 0 ? a : b); };
                default:
                    throw new NotImplementedException($"Unkown aggregated function ${func}.");
            }
        }
    }
}
