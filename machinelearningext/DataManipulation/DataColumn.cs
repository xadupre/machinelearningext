// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements a dense column container.
    /// </summary>
    public class DataColumn<DType> : IDataColumn, IEquatable<DataColumn<DType>>, IEnumerable<DType>
        where DType : IEquatable<DType>, IComparable<DType>
    {
        #region members and easy functions

        /// <summary>
        /// Data for the column.
        /// </summary>
        DType[] _data;

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public IDataColumn Copy()
        {
            var res = new DataColumn<DType>(Length);
            Array.Copy(_data, res._data, Length);
            return res;
        }

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        public IDataColumn Copy(IEnumerable<int> rows)
        {
            var arows = rows.ToArray();
            var res = new DataColumn<DType>(arows.Length);
            for (int i = 0; i < arows.Length; ++i)
                res._data[i] = _data[arows[i]];
            return res;
        }

        /// <summary>
        /// Number of elements.
        /// </summary>
        public int Length => (_data == null ? 0 : _data.Length);

        /// <summary>
        /// Get a pointer on the raw data.
        /// </summary>
        public DType[] Data => _data;

        public object Get(int row) { return _data[row]; }
        public void Set(int row, object value)
        {
            DType dt;
            ObjectConversion.Convert(ref value, out dt);
            Set(row, dt);
        }

        /// <summary>
        /// Returns type data kind.
        /// </summary>
        public DataKind Kind => SchemaHelper.GetKind<DType>();

        public IEnumerator<DType> GetEnumerator() { foreach (var v in _data) yield return v; }
        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }

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

        /// <summary>
        /// Changes all values.
        /// </summary>
        public void Set(object value)
        {
            DType dt;
            ObjectConversion.Convert(ref value, out dt);
            for (var row = 0; row < Length; ++row)
                _data[row] = dt;
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<bool> rows, object value)
        {
            var irow = 0;
            foreach (var row in rows)
            {
                if (row)
                    Set(irow, value);
                ++irow;
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<int> rows, object value)
        {
            foreach (var row in rows)
                Set(row, value);
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<bool> rows, IEnumerable<object> values)
        {
            var iter = values.GetEnumerator();
            var irow = 0;
            foreach (var row in rows)
            {
                iter.MoveNext();
                if (row)
                    Set(irow, iter.Current);
                ++irow;
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<int> rows, IEnumerable<object> values)
        {
            var iter = values.GetEnumerator();
            foreach (var row in rows)
            {
                iter.MoveNext();
                Set(row, iter.Current);
            }
        }

        #endregion

        #region linq

        public IEnumerable<bool> Filter<DType2>(Func<DType2, bool> predicate)
        {
            return (_data as DType2[]).Select(c => predicate(c));
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
