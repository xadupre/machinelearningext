// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements a dense column container.
    /// </summary>
    public class DataColumn<DType> : IDataColumn, IEquatable<DataColumn<DType>>
        where DType : IEquatable<DType>
    {
        #region memeber

        /// <summary>
        /// Data for the column.
        /// </summary>
        DType[] _data;

        /// <summary>
        /// Number of elements.
        /// </summary>
        public int Length => (_data == null ? 0 : _data.Length);

        /// <summary>
        /// Get a pointer on the raw data.
        /// </summary>
        public DType[] Data => _data;

        public object Get(int row) { return _data[row]; }
        public void Set(int row, object value) { Set(row, (DType)value); }

        /// <summary>
        /// Returns type data kind.
        /// </summary>
        public DataKind Kind => SchemaHelper.GetKind<DType>();

        #endregion

        #region constructor

        /// <summary>
        /// Builds the columns.
        /// </summary>
        /// <param name="nb"></param>
        public DataColumn(int nb)
        {
            _data = new DType[nb];
        }

        /// <summary>
        /// Changes the value at a specific row.
        /// </summary>
        public void Set(int row, DType value)
        {
            _data[row] = value;
        }

        #endregion

        #region getter and comparison

        /// <summary>
        /// Creates a getter on the column. The getter returns the element at
        /// cursor.Position.
        /// </summary>
        public ValueGetter<DType2> GetGetter<DType2>(IRowCursor cursor)
        {
            var _data2 = _data as DType2[];
            return (ref DType2 value) => { value = _data2[cursor.Position]; };
        }

        public bool Equals(IDataColumn c)
        {
            var obj = c as DataColumn<DType>;
            if (obj == null)
                return false;
            return Equals(obj);
        }

        public bool Equals(DataColumn<DType> c)
        {
            if (Length != c.Length)
                return false;
            for (int i = 0; i < Length; ++i)
                if (!_data[i].Equals(c._data[i]))
                    return false;
            return true;
        }

        #endregion
    }
}
