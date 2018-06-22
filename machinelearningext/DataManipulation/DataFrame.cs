// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements a DataFrame based on a IDataView from ML.net.
    /// </summary>
    public class DataFrame : IDataView, IDataFrameView, IEquatable<DataFrame>
    {
        #region members

        IHost _host;
        DataContainer _data;

        #endregion

        public DataFrame(IHost host)
        {
            _host = host;
            _data = new DataContainer(_host);
        }

        #region IDataView API

        /// <summary>
        /// Can shuffle the data.
        /// </summary>
        public bool CanShuffle { get { return true; } }

        /// <summary>
        /// Returns the number of rows
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            return _data.Length;
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return _data.GetRowCursor(needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _data.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        public ISchema Schema => _data.Schema;

        #endregion

        #region DataFrame

        public Tuple<int, int> Shape => _data.Shape;

        /// <summary>
        /// Adds a new column.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="kind">column type</param>
        /// <param name="length">overwrites length</param>
        public int AddColumn(string name, DataKind kind, int? length)
        {
            return _data.AddColumn(name, kind, length);
        }

        #endregion

        #region IO

        public string NameType(int col) { return _data.NameType(col); }

        /// <summary>
        /// Saves the dataframe as a file.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="encoding">encoding</param>
        public void ToCsv(string filename, string sep = ",", bool header = true, Encoding encoding = null)
        {
            ViewToCsv(_host, this, filename, sep: sep, header: header, encoding: encoding);
        }

        /// <summary>
        /// Saves the dataframe as a file.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="encoding">encoding</param>
        public static void ViewToCsv(IHost host, IDataView view, string filename, string sep = ",", bool header = true, Encoding encoding = null)
        {
            var saver = new TextSaver(host, new TextSaver.Arguments()
            {
                Separator = sep,
                OutputSchema = false,
                OutputHeader = header
            });
            var columns = new int[view.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = i;
            using (var fs = new StreamWriter(filename, false, encoding ?? Encoding.ASCII))
                saver.SaveData(fs.BaseStream, view, columns);
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="host">host</param>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <param name="useThreads">specific to TextLoader</param>
        /// <returns>TextLoader</returns>
        public static TextLoader ReadCsvToTextLoader(IHost host, string filename,
                                        char sep = ',', bool header = true,
                                        string[] names = null,
                                        DataKind?[] dtypes = null,
                                        int nrows = -1,
                                        int guess_rows = 10,
                                        Encoding encoding = null,
                                        bool useThreads = true)
        {
            var df = ReadCsv(host, filename, sep: sep, header: header, names: names, dtypes: dtypes,
                             nrows: guess_rows, guess_rows: guess_rows, encoding: encoding);
            var sch = df.Schema;
            var cols = new TextLoader.Column[sch.ColumnCount];
            for (int i = 0; i < cols.Length; ++i)
                cols[i] = TextLoader.Column.Parse(df.NameType(i));

            var args = new TextLoader.Arguments()
            {
                AllowQuoting = false,
                Separator = string.Format("{0}", sep),
                Column = cols,
                TrimWhitespace = true,
                UseThreads = useThreads,
                HasHeader = header,
                MaxRows = nrows > 0 ? (int?)nrows : null
            };
            return new TextLoader(host, args, new MultiFileSource(filename));
        }

        /// <summary>
        /// Reads a string as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="host">host</param>
        /// <param name="content">data as single string</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadStr(IHost host, string content,
                                    char sep = ',', bool header = true,
                                    string[] names = null,
                                    DataKind?[] dtypes = null,
                                    int nrows = -1,
                                    int guess_rows = 10)
        {
            return ReadStream(host, () => new StreamReader(new MemoryStream(Encoding.UTF8.GetBytes(content))),
                              sep: sep, header: header, names: names, dtypes: dtypes, nrows: nrows,
                              guess_rows: guess_rows);
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="host">host</param>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadCsv(IHost host, string filename,
                                char sep = ',', bool header = true,
                                string[] names = null,
                                DataKind?[] dtypes = null,
                                int nrows = -1,
                                int guess_rows = 10,
                                Encoding encoding = null)
        {
            return ReadStream(host, () => new StreamReader(filename, encoding ?? Encoding.ASCII),
                              sep: sep, header: header, names: names, dtypes: dtypes, nrows: nrows,
                              guess_rows: guess_rows);
        }

        public delegate StreamReader FunctionCreateStreamReader();

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="host">host</param>
        /// <param name="createStream">function which creates a stream</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadStream(IHost host, FunctionCreateStreamReader createStream,
                                char sep = ',', bool header = true,
                                string[] names = null,
                                DataKind?[] dtypes = null,
                                int nrows = -1,
                                int guess_rows = 10)
        {
            var lines = new List<string[]>();
            int rowline = 0;

            // First pass: schema and number of rows.
            using (var st = createStream())
            {
                string line = st.ReadLine();
                int nbline = 0;
                while (line != null && (nrows == -1 || rowline < nrows))
                {
                    var spl = line.Split(sep);
                    if (header && nbline == 0)
                    {
                        if (names == null)
                            names = spl;
                    }
                    else
                    {
                        ++rowline;
                        if (lines.Count < guess_rows)
                            lines.Add(spl);
                    }
                    ++nbline;
                    line = st.ReadLine();
                }
            }

            if (lines.Count == 0)
                throw new FormatException("File is empty.");
            int numCol = lines.Select(c => c.Length).Max();
            var df = new DataFrame(host);

            // Guesses types and adds columns.
            for (int i = 0; i < numCol; ++i)
            {
                var kind = GuessKind(i, lines);
                df.AddColumn(names[i], dtypes != null && i < dtypes.Length && dtypes[i].HasValue ? dtypes[i].Value : kind, rowline);
            }

            // Fills values.
            using (var st = createStream())
            {
                string line = st.ReadLine();
                int nbline = 0;
                rowline = 0;
                while (line != null && (nrows == -1 || rowline < nrows))
                {
                    var spl = line.Split(sep);
                    if (header && nbline == 0)
                    {
                        // Skips.
                    }
                    else
                    {
                        df.FillValues(rowline, spl);
                        ++rowline;
                    }
                    ++nbline;
                    line = st.ReadLine();
                }
            }
            return df;
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="host">host</param>
        /// <param name="view">IDataView</param>
        /// <param name="sep">column separator</param>
        /// <param name="nrows">number of rows to read</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadView(IHost host, IDataView view, int nrows = -1)
        {
            var df = new DataFrame(host);
            df.FillValues(view, nrows: nrows);
            return df;
        }

        public void FillValues(IDataView view, int nrows = -1)
        {
            _data.FillValues(view, nrows: nrows);
        }

        static DataKind GuessKind(int col, List<string[]> read)
        {
            DataKind res = DataKind.TX;
            int nbline = 0;
            foreach (var line in read)
            {
                if (col >= line.Length)
                    throw new FormatException(string.Format("Line {0} has less column than expected.", nbline + 1));
                var val = line[col];

                try
                {
                    var v = bool.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.BL, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.BL, res);
                        continue;
                    }
                }

                try
                {
                    var v = int.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.I4, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.I4, res);
                        continue;
                    }
                }

                try
                {
                    var v = Int64.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.I8, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.I8, res);
                        continue;
                    }
                }

                try
                {
                    var v = float.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.R4, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.R4, res);
                        continue;
                    }
                }

                try
                {
                    var v = double.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.R8, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.R8, res);
                        continue;
                    }
                }

                res = DetermineDataKind(nbline == 0, DataKind.TX, res);
                ++nbline;
            }
            return res;
        }

        static DataKind DetermineDataKind(bool first, DataKind suggested, DataKind previous)
        {
            if (first)
                return suggested;
            else
                return MaxKind(suggested, previous);
        }

        static DataKind MaxKind(DataKind a, DataKind b)
        {
            if (a == DataKind.TX || b == DataKind.TX)
                return DataKind.TX;
            if (a == DataKind.R8 || b == DataKind.R8)
                return DataKind.R8;
            if (a == DataKind.R4 || b == DataKind.R4)
                return DataKind.R4;
            if (a == DataKind.I8 || b == DataKind.I8)
                return DataKind.I8;
            if (a == DataKind.I4 || b == DataKind.I4)
                return DataKind.I4;
            if (a == DataKind.BL || b == DataKind.BL)
                return DataKind.BL;
            return DataKind.TX;
        }

        /// <summary>
        /// Changes the value for an entire row.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="values"></param>
        public void FillValues(int row, string[] values)
        {
            _data.FillValues(row, values);
        }

        #endregion

        #region pandas API (slow)

        public Iloc iloc => new Iloc(this);

        public class Iloc
        {
            DataFrame _parent;
            public Iloc(DataFrame parent)
            {
                _parent = parent;
            }

            public object this[int row, int col]
            {
                get
                {
                    return _parent._data[row, col];
                }
                set
                {
                    _parent._data[row, col] = value;
                }
            }
        }

        #endregion

        #region assert

        public static bool operator ==(DataFrame df1, DataFrame df2)
        {
            return df1._data == df2._data;
        }

        public static bool operator !=(DataFrame df1, DataFrame df2)
        {
            return df1._data != df2._data;
        }

        public bool Equals(DataFrame df)
        {
            return _data.Equals(df._data);
        }

        public override bool Equals(object o)
        {
            var df = o as DataFrame;
            if (df == null)
                return false;
            return Equals(df);
        }

        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
