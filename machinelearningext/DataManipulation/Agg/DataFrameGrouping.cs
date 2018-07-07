// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements grouping functions for dataframe.
    /// </summary>
    public static class DataFrameGrouping
    {
        public static IEnumerable<KeyValuePair<TKey, DataFrameViewGroup>> GroupBy<TKey>(
                                IDataFrameView df, int[] order, TKey[] keys, int[] columns,
                                Func<TKey, DataFrameGroupKey[]> func)
            where TKey : IEquatable<TKey>
        {
            TKey last = default(TKey);
            List<int> subrows = new List<int>();
            foreach (var pos in order)
            {
                var cur = keys[pos];
                if (last == null || cur.Equals(last))
                    subrows.Add(pos);
                else if (subrows.Any())
                {
                    yield return new KeyValuePair<TKey, DataFrameViewGroup>(last,
                                    new DataFrameViewGroup(func(last), df.Source, subrows.ToArray(), columns));
                    subrows.Clear();
                    subrows.Add(pos);
                }
                last = cur;
            }
            if (subrows.Any())
                yield return new KeyValuePair<TKey, DataFrameViewGroup>(last,
                            new DataFrameViewGroup(func(last), df.Source, subrows.ToArray(), columns));
        }

        public static DataFrameViewGroupResults<TImutKey> GroupBy<TMutKey, TImutKey>(
                            IDataFrameView df, int[] rows, int[] columns, IEnumerable<int> cols, bool sort,
                            MultiGetterAt<TMutKey> getter,
                            Func<TMutKey, TImutKey> conv,
                            Func<TImutKey, DataFrameGroupKey[]> conv2)
            where TMutKey : ITUple, new()
            where TImutKey : IComparable<TImutKey>, IEquatable<TImutKey>
        {
            var icols = cols.ToArray();
            int[] order = rows == null ? rows.Select(c => c).ToArray() : Enumerable.Range(0, df.Length).ToArray();
            var keys = df.EnumerateItems(icols, true, rows, getter).Select(c => conv(c)).ToArray();
            if (sort)
                DataFrameSorting.TSort(df, ref order, keys, true);
            var iter = GroupBy(df, order, keys, columns, conv2);
            return new DataFrameViewGroupResults<TImutKey>(iter);
        }
    }
}
