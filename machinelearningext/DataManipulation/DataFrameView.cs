// See the LICENSE file in the project root for more information.

using System;
using System.IO;
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

        public int[] ALL { get { return null; } }
        public int Length { get { return _src.Length; } }

        /// <summary>
        /// Initializes a view on a dataframe.
        /// </summary>
        public DataFrameView(IDataFrameView src, IEnumerable<int> rows, IEnumerable<int> columns)
        {
            _src = src;
            _rows = rows == null ? Enumerable.Range(0, src.Length).ToArray() : rows.ToArray();
            _columns = columns == null ? Enumerable.Range(0, src.Schema.ColumnCount).ToArray() : columns.ToArray();
            _schema = new DataFrameViewSchema(src.Schema, _columns);
        }

        #region IDataView API

        public ISchema Schema => _schema;
        public int ColumnCount => _columns == null ? _src.ColumnCount : _columns.Length;

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

        public IRowCursor GetRowCursor(int[] rows, int[] columns, Func<int, bool> needCol, IRandom rand = null)
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

        /// <summary>
        /// Converts the data frame into a string.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            using (var stream = new MemoryStream())
            {
                DataFrame.ViewToCsv(this, stream);
                stream.Position = 0;
                using (var reader = new StreamReader(stream))
                    return reader.ReadToEnd().Replace("\r", "").TrimEnd(new char[] { '\n' });
            }
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

        /// <summary>
        /// Drops some columns.
        /// Data is not copied.
        /// </summary>
        public DataFrameView Drop(IEnumerable<string> colNames)
        {
            var idrop = new HashSet<int>(colNames.Select(c => { int col; Schema.TryGetColumnIndex(c, out col); return col; }));
            var ikeep = Enumerable.Range(0, ColumnCount).Where(c => !idrop.Contains(c));
            return new DataFrameView(_src, _rows, ikeep);
        }

        #endregion

        #region operators

        static int[] ComposeArrayInt(int[] a1, int[] a2)
        {
            var res = new int[a2.Length];
            for (int i = 0; i < a2.Length; ++i)
                res[i] = a1[a2[i]];
            return res;
        }

        /// <summary>
        /// Retrieves a column by its name.
        /// </summary>
        public NumericColumn GetColumn(string colname, int[] rows = null)
        {
            return _src.GetColumn(colname, rows == null ? _rows : ComposeArrayInt(_rows, rows));
        }

        /// <summary>
        /// Retrieves a column by its position.
        /// </summary>
        public NumericColumn GetColumn(int col, int[] rows = null)
        {
            return _src.GetColumn(col, rows == null ? _rows : ComposeArrayInt(_rows, rows));
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public NumericColumn this[string colname]
        {
            get { return GetColumn(colname); }
        }

        #endregion

        #region loc / iloc

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public Iloc iloc => new Iloc(this);

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public class Iloc
        {
            DataFrameView _parent;

            public Iloc(DataFrameView parent)
            {
                _parent = parent;
            }

            DataFrame AsDataFrame()
            {
                var parent = _parent._src as DataFrame;
                if (parent is null)
                    throw new NotImplementedException(string.Format("Not implement for type '{0}'.", _parent._src.GetType()));
                return parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, int col]
            {
                get { return AsDataFrame().iloc[_parent._rows[row], _parent._columns[col]]; }
                set { AsDataFrame().iloc[_parent._rows[row], col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, int col]
            {
                set { AsDataFrame().iloc[rows, col] = value; }
            }
        }

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public Loc loc => new Loc(this);

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public class Loc
        {
            DataFrameView _parent;

            public Loc(DataFrameView parent)
            {
                _parent = parent;
            }

            DataFrame AsDataFrame()
            {
                var parent = _parent._src as DataFrame;
                if (parent is null)
                    throw new NotImplementedException(string.Format("Not implement for type '{0}'.", _parent._src.GetType()));
                return parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, string col]
            {
                get { return AsDataFrame().loc[_parent._rows[row], col]; }
                set { AsDataFrame().loc[_parent._rows[row], col] = value; }
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[string col]
            {
                set { AsDataFrame().loc[col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<int> rows, string col]
            {
                get
                {
                    int icol;
                    _parent._src.Schema.TryGetColumnIndex(col, out icol);
                    return new DataFrameView(_parent._src, rows.Select(c => _parent._rows[c]), new[] { icol });
                }
                set { AsDataFrame().loc[rows.Select(c => _parent._rows[c]), col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, string col]
            {
                set { AsDataFrame().loc[Enumerable.Zip(_parent._rows, rows, (i, b) => b ? -1 : i).Where(c => c >= 0), col] = value; }
            }
        }

        #endregion
    }
}
