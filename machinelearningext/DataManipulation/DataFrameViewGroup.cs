// See the LICENSE file in the project root for more information.

using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    public struct DataFrameGroupKey
    {
        public string Key;
        public DataKind Kind;
        public object Value;
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
