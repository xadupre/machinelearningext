// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Helper for numeric purpose.
    /// </summary>
    public static class NumericHelper
    {
        public static bool AlmostEquals(double exp, double res, double precision = 1e-7)
        {
            if (exp == 0)
                return Math.Abs(res) < precision;
            else
            {
                double delta = exp - res;
                double rel = (Math.Abs(delta) / Math.Abs(exp));
                return rel < precision;
            }
        }

        public static bool AlmostEquals(float exp, float res, float precision = 1e-7f)
        {
            if (exp == 0)
                return Math.Abs(res) < precision;
            else
            {
                float delta = exp - res;
                float rel = (Math.Abs(delta) / Math.Abs(exp));
                return rel < precision;
            }
        }

        public static bool AlmostEquals(IEnumerable<double> exp, IEnumerable<double> res, double precision = 1e-7)
        {
            return Enumerable.All(Enumerable.Zip(exp, res, (a, b) => AlmostEquals(a, b, precision)), c => c);
        }

        public static bool AlmostEquals(IEnumerable<float> exp, IEnumerable<float> res, float precision = 1e-7f)
        {
            return Enumerable.All(Enumerable.Zip(exp, res, (a, b) => AlmostEquals(a, b, precision)), c => c);
        }
    }
}
