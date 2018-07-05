// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;


namespace Microsoft.ML.Ext.DataManipulation
{
    public static class DataFrameSorting
    {
        static void Sort<T>(IDataFrameView df, ref int[] order, T[] keys, bool ascending)
            where T:IComparable<T>
        {
            if (order == null)
            {
                order = new int[df.Length];
                for (int i = 0; i < order.Length; ++i)
                    order[i] = i;
            }
            if (ascending)
                Array.Sort(order, (x, y) => keys[x].CompareTo(keys[y]));
            else
                Array.Sort(order, (x, y) => -keys[x].CompareTo(keys[y]));
        }

        public static void Sort<T1>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var keys = df.EnumerateItems<T1>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            Sort(df, ref order, keys, ascending);
        }

        public static void Sort<T1, T2>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var keys = df.EnumerateItems<T1, T2>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            Sort(df, ref order, keys, ascending);
        }

        public static void Sort<T1, T2, T3>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var keys = df.EnumerateItems<T1, T2, T3>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            Sort(df, ref order, keys, ascending);
        }
    }
}
