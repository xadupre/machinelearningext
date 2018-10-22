// See the LICENSE file in the project root for more information.

using System;
using Scikit.ML.DataManipulation;


namespace Scikit.ML.TestHelper
{
    public static class DataFrameHelperTest
    {
        public static DataFrame CreateDataFrameWithAllTypes()
        {
            var df = new DataFrame();
            df.AddColumn("cbool", new bool[] { true, false });
            df.AddColumn("cint", new[] { 1, 0 });
            df.AddColumn("cuint", new uint[] { 3, 4 });
            df.AddColumn("cint64", new Int64[] { 3, 4 });
            df.AddColumn("cfloat", new[] { 6f, 7f });
            df.AddColumn("cdouble", new[] { 6.5, 7.5 });
            df.AddColumn("ctext", new[] { "t1", "t2" });

            df.AddColumn("vbool", new bool[][] { new bool[] { true, false, true }, new bool[] { true, false, true } });
            df.AddColumn("vint", new int[][] { new int[] { 1, 0 }, new int[] { 2, 3 } });
            df.AddColumn("vuint", new uint[][] { new uint[] { 3, 4 }, new uint[] { 5, 6 } });
            df.AddColumn("vint64", new Int64[][] { new Int64[] { 3, 4 }, new Int64[] { 5, 6 } });
            df.AddColumn("vfloat", new float[][] { new float[] { 6f, 7f }, new float[] { 8f, 9f, 10f } });
            df.AddColumn("vdouble", new double[][] { new double[] { 6.6, 7.6 }, new double[] { 8.6, 9.6, 10.6 } });
            df.AddColumn("vtext", new string[][] { new string[] { "t1", "t2" }, new string[] { "t5", "t6", "t8" } });

            return df;
        }
    }
}
