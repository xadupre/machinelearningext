// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Wraps a column and adds numerical operators.
    /// </summary>
    public class NumericColumn : IDataColumn
    {
        protected IDataColumn _column;

        public NumericColumn(IDataColumn column)
        {
            _column = column;
        }

        public IDataColumn Column { get { return _column; } }
        public int Length => _column.Length;
        public DataKind Kind => _column.Kind;
        public object Get(int row) => _column.Get(row);
        public void Set(int row, object value) { _column.Set(row, value); }
        public ValueGetter<DType> GetGetter<DType>(IRowCursor cursor) => _column.GetGetter<DType>(cursor);
        public bool Equals(IDataColumn col) => _column.Equals(col);

        public virtual DType[] GetData<DType>()
        {
            throw new NotImplementedException("This function must be overwritten.");
        }

        public static NumericColumn operator +(NumericColumn c1, NumericColumn c2)
        {
            return DataFrameOperationHelper.Addition(c1, c2);
        }

        public static NumericColumn operator +(NumericColumn c1, int value)
        {
            return DataFrameOperationHelper.Addition(c1, (DvInt4)value);
        }

        public static NumericColumn operator +(NumericColumn c1, DvInt4 value)
        {
            return DataFrameOperationHelper.Addition(c1, value);
        }

        public static NumericColumn operator +(NumericColumn c1, float value)
        {
            return DataFrameOperationHelper.Addition(c1, value);
        }

        public static NumericColumn operator +(NumericColumn c1, double value)
        {
            return DataFrameOperationHelper.Addition(c1, value);
        }

        public static NumericColumn operator +(NumericColumn c1, DvText value)
        {
            return DataFrameOperationHelper.Addition(c1, value);
        }

        public static NumericColumn operator +(NumericColumn c1, string value)
        {
            return DataFrameOperationHelper.Addition(c1, new DvText(value));
        }
    }
}
