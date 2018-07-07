// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.DataManipulation
{
    public struct DataFrameGroupKey
    {
        public string Key;
        public DataKind Kind;
        public object Value;

        public static DataFrameGroupKey[] Create<T1>(string[] name, ImmutableTuple<T1> value)
            where T1 : IComparable<T1>, IEquatable<T1>
        {
            var res = new DataFrameGroupKey[1];
            res[0].Key = name[0];
            res[0].Kind = SchemaHelper.GetKind<T1>();
            res[0].Value = (object)value.Item1;
            return res;
        }

        public static DataFrameGroupKey[] Create<T1, T2>(string[] name, ImmutableTuple<T1, T2> value)
            where T1 : IComparable<T1>, IEquatable<T1>
            where T2 : IComparable<T2>, IEquatable<T2>
        {
            var res = new DataFrameGroupKey[2];
            res[0].Key = name[0];
            res[0].Kind = SchemaHelper.GetKind<T1>();
            res[0].Value = (object)value.Item1;
            res[1].Key = name[1];
            res[1].Kind = SchemaHelper.GetKind<T2>();
            res[1].Value = (object)value.Item2;
            return res;
        }

        public static DataFrameGroupKey[] Create<T1, T2, T3>(string[] name, ImmutableTuple<T1, T2, T3> value)
            where T1 : IComparable<T1>, IEquatable<T1>
            where T2 : IComparable<T2>, IEquatable<T2>
            where T3 : IComparable<T3>, IEquatable<T3>
        {
            var res = new DataFrameGroupKey[1];
            res[0].Key = name[0];
            res[0].Kind = SchemaHelper.GetKind<T1>();
            res[0].Value = (object)value.Item1;
            res[1].Key = name[1];
            res[1].Kind = SchemaHelper.GetKind<T2>();
            res[1].Value = (object)value.Item2;
            res[2].Key = name[2];
            res[2].Kind = SchemaHelper.GetKind<T3>();
            res[2].Value = (object)value.Item3;
            return res;
        }
    }

    /// <summary>
    /// View produced by method GroupBy.
    /// </summary>
    public class DataFrameViewGroup : DataFrameView
    {
        DataFrameGroupKey[] _keys;

        public string[] ColumnsKey => _keys.Select(c => c.Key).ToArray();
        public DataFrameGroupKey[] Keys => _keys;

        public DataFrameViewGroup(DataFrameGroupKey[] keys, IDataFrameView src, IEnumerable<int> rows, IEnumerable<int> columns) : base(src, rows, columns)
        {
            _keys = keys;
        }
    }
}
