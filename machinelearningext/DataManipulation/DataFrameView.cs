// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    public class DataFrameView : IDataFrameView
    {
        IDataFrameView _src;
        int[] _rows;
        int[] _columns;
        ISchema _schema;

        /// <summary>
        /// Initializes a view on a dataframe.
        /// </summary>
        public DataFrameView(IDataFrameView src, IEnumerable<int> rows, IEnumerable<int> columns)
        {
            _src = src;
            _rows = rows.ToArray();
            _columns = columns.ToArray();
            _schema = new DataFrameViewSchema(src.Schema, _columns);
        }

        #region IDataView API

        public ISchema Schema => _schema;

        /// <summary>
        /// Can shuffle the data.
        /// </summary>
        public bool CanShuffle { get { return _src.CanShuffle; } }

        /// <summary>
        /// Returns the number of rows. lazy is unused as the data is stored in memory.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            return _rows.Length;
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return _src.GetRowCursor(_rows, _columns, needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _src.GetRowCursorSet(_rows, _columns, out consolidator, needCol, n, rand);
        }

        public IRowCursor GetRowCursor(int []rows, int[] columns, Func<int, bool> needCol, IRandom rand = null)
        {
            throw Contracts.ExceptNotSupp("Not applicable here, consider building a DataFrameView.");
        }

        public IRowCursor[] GetRowCursorSet(int[] rows, int[] columns, out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            throw Contracts.ExceptNotSupp("Not applicable here, consider building a DataFrameView.");
        }

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        public DataFrame Copy()
        {
            return _src.Copy(_rows, _columns);
        }

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        public DataFrame Copy(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            var rows2 = rows.Select(i => _rows[i]);
            var columns2 = columns.Select(i => _columns[i]);
            return _src.Copy(rows2, columns2);
        }

        #endregion

        #region DataFrame

        /// <summary>
        /// Returns the shape of the dataframe (number of rows, number of columns).
        /// </summary>
        public Tuple<int, int> Shape => new Tuple<int, int>(_rows.Length, _columns.Length);

        /// <summary>
        /// A view cannot be modified by adding a column.
        /// </summary>
        public int AddColumn(string name, DataKind kind, int? length)
        {
            throw new DataFrameViewException("A column cannot be added to a DataFrameView.");
        }

        /// <summary>
        /// A view cannot be modified by adding a column.
        /// It must be the same for all columns.
        /// </summary>
        public int AddColumn(string name, IDataColumn values)
        {
            throw new DataFrameViewException("A column cannot be added to a DataFrameView.");
        }

        /// <summary>
        /// Compares two view. First converts them into a DataFrame.
        /// </summary>
        public bool Equals(IDataFrameView dfv)
        {
            return Copy().Equals(dfv.Copy());
        }

        #endregion
    }
}
