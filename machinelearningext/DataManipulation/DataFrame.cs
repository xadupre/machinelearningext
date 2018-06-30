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
    public class DataFrame : IDataFrameView
    {
        #region members

        DataContainer _data;
        bool _shuffle;

        /// <summary>
        /// Can shuffle the data.
        /// </summary>
        public bool CanShuffle => _shuffle;


        public int[] ALL { get { return null; } }

        #endregion

        #region constructor

        /// <summary>
        /// Initializes an empty dataframe.
        /// </summary>
        /// <param name="shuffle">The dataframe can be shuffled.</param>
        public DataFrame(bool shuffle = true)
        {
            _data = new DataContainer();
            _shuffle = shuffle;
        }

        public void SetShuffle(bool shuffle)
        {
            _shuffle = shuffle;
        }

        #endregion

        #region IDataView API

        /// <summary>
        /// Returns the number of rows. lazy is unused as the data is stored in memory.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            return _data.Length;
        }

        public int Length { get { return _data.Length; } }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return _data.GetRowCursor(needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _data.GetRowCursorSet(out consolidator, needCol, n, rand);
        }

        public IRowCursor GetRowCursor(int[] rows, int[] columns, Func<int, bool> needCol, IRandom rand = null)
        {
            return _data.GetRowCursor(rows, columns, needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(int[] rows, int[] columns, out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _data.GetRowCursorSet(rows, columns, out consolidator, needCol, n, rand);
        }

        /// <summary>
        /// Returns the schema of the dataframe, used schema used for IDataView.
        /// </summary>
        public ISchema Schema => _data.Schema;

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        public DataFrame Copy()
        {
            var df = new DataFrame();
            df._data = _data.Copy();
            return df;
        }

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        public DataFrame Copy(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            var df = new DataFrame();
            df._data = _data.Copy(rows, columns);
            return df;
        }

        #endregion

        #region DataFrame

        /// <summary>
        /// Returns the shape of the dataframe (number of rows, number of columns).
        /// </summary>
        public Tuple<int, int> Shape => _data.Shape;

        /// <summary>
        /// Adds a new column. The length must be specified for the first column.
        /// It must be the same for all columns.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="kind">column type</param>
        /// <param name="length">length is needed for the first column to allocated space</param>
        public int AddColumn(string name, DataKind kind, int? length)
        {
            return _data.AddColumn(name, kind, length);
        }

        /// <summary>
        /// Adds a new column. The length must be specified for the first column.
        /// It must be the same for all columns.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="values">new column</param>
        public int AddColumn(string name, IDataColumn values)
        {
            return _data.AddColumn(name, values.Kind, values.Length, values);
        }

        public int AddColumn(string name, DvBool[] values) { return AddColumn(name, new DataColumn<DvBool>(values)); }
        public int AddColumn(string name, bool[] values)
        {
            var buf = new DvBool[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = values[i];
            return AddColumn(name, new DataColumn<DvBool>(buf));
        }
        public int AddColumn(string name, DvInt4[] values) { return AddColumn(name, new DataColumn<DvInt4>(values)); }
        public int AddColumn(string name, int[] values)
        {
            var buf = new DvInt4[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = values[i];
            return AddColumn(name, new DataColumn<DvInt4>(buf));
        }
        public int AddColumn(string name, DvInt8[] values) { return AddColumn(name, new DataColumn<DvInt8>(values)); }
        public int AddColumn(string name, Int64[] values)
        {
            var buf = new DvInt8[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = values[i];
            return AddColumn(name, new DataColumn<DvInt8>(buf));
        }
        public int AddColumn(string name, uint[] values) { return AddColumn(name, new DataColumn<uint>(values)); }
        public int AddColumn(string name, float[] values) { return AddColumn(name, new DataColumn<float>(values)); }
        public int AddColumn(string name, double[] values) { return AddColumn(name, new DataColumn<double>(values)); }
        public int AddColumn(string name, DvText[] values) { return AddColumn(name, new DataColumn<DvText>(values)); }
        public int AddColumn(string name, string[] values)
        {
            var buf = new DvText[values.Length];
            for (int i = 0; i < values.Length; ++i)
                buf[i] = new DvText(values[i]);
            return AddColumn(name, new DataColumn<DvText>(buf));
        }

        #endregion

        #region IO

        /// <summary>
        /// Returns the name and the type of a column such as
        /// <pre>name:type:index</pre>.
        /// </summary>
        public string NameType(int col) { return _data.NameType(col); }

        /// <summary>
        /// Converts the data frame into a string.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            using (var stream = new MemoryStream())
            {
                ViewToCsv(this, stream, silent: true);
                stream.Position = 0;
                using (var reader = new StreamReader(stream))
                    return reader.ReadToEnd().Replace("\r", "").TrimEnd(new char[] { '\n' });
            }
        }

        /// <summary>
        /// Saves the dataframe as a file.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="encoding">encoding</param>
        /// <param name="silent">Suppress any info output (not warnings or errors)</param>
        public void ToCsv(string filename, string sep = ",", bool header = true, Encoding encoding = null, bool silent = false)
        {
            ViewToCsv(this, filename, sep: sep, header: header, encoding: encoding, silent: silent);
        }

        /// <summary>
        /// Saves the dataframe as a file.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="encoding">encoding</param>
        /// <param name="silent">Suppress any info output (not warnings or errors)</param>
        public static void ViewToCsv(IDataView view, string filename, string sep = ",",
                                     bool header = true, Encoding encoding = null, bool silent = false)
        {
            using (var fs = new StreamWriter(filename, false, encoding ?? Encoding.ASCII))
                ViewToCsv(view, fs.BaseStream, sep: sep, header: header, silent: silent);
        }

        /// <summary>
        /// Saves the dataframe in a stream as text format.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">add header</param>
        /// <param name="silent">Suppress any info output (not warnings or errors)</param>
        public static void ViewToCsv(IDataView view, Stream st, string sep = ",", bool header = true,
                                     bool silent = false)
        {
            var env = new TlcEnvironment();
            var saver = new TextSaver(env, new TextSaver.Arguments()
            {
                Separator = sep,
                OutputSchema = false,
                OutputHeader = header,
                Silent = silent
            });
            var columns = new int[view.Schema.ColumnCount];
            for (int i = 0; i < columns.Length; ++i)
                columns[i] = i;
            saver.SaveData(st, view, columns);
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
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
        public static TextLoader ReadCsvToTextLoader(string filename,
                                        char sep = ',', bool header = true,
                                        string[] names = null,
                                        DataKind?[] dtypes = null,
                                        int nrows = -1,
                                        int guess_rows = 10,
                                        Encoding encoding = null,
                                        bool useThreads = true,
                                        IHost host = null)
        {
            var df = ReadCsv(filename, sep: sep, header: header, names: names, dtypes: dtypes,
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
            if (host == null)
                host = new TlcEnvironment().Register("TextLoader");
            return new TextLoader(host, args, new MultiFileSource(filename));
        }

        /// <summary>
        /// Reads a string as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="content">data as single string</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadStr(string content,
                                    char sep = ',', bool header = true,
                                    string[] names = null,
                                    DataKind?[] dtypes = null,
                                    int nrows = -1,
                                    int guess_rows = 10)
        {
            return ReadStream(() => new StreamReader(new MemoryStream(Encoding.UTF8.GetBytes(content))),
                              sep: sep, header: header, names: names, dtypes: dtypes, nrows: nrows,
                              guess_rows: guess_rows);
        }

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="filename">filename</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadCsv(string filename,
                                char sep = ',', bool header = true,
                                string[] names = null,
                                DataKind?[] dtypes = null,
                                int nrows = -1,
                                int guess_rows = 10,
                                Encoding encoding = null)
        {
            return ReadStream(() => new StreamReader(filename, encoding ?? Encoding.ASCII),
                              sep: sep, header: header, names: names, dtypes: dtypes, nrows: nrows,
                              guess_rows: guess_rows);
        }

        public delegate StreamReader FunctionCreateStreamReader();

        /// <summary>
        /// Reads a text file as a IDataView.
        /// Follows pandas API.
        /// </summary>
        /// <param name="createStream">function which creates a stream</param>
        /// <param name="sep">column separator</param>
        /// <param name="header">has a header or not</param>
        /// <param name="names">column names (can be empty)</param>
        /// <param name="dtypes">column types (can be empty)</param>
        /// <param name="nrows">number of rows to read</param>
        /// <param name="guess_rows">number of rows used to guess types</param>
        /// <param name="encoding">text encoding</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadStream(FunctionCreateStreamReader createStream,
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
            var df = new DataFrame();

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
        /// <param name="view">IDataView</param>
        /// <param name="sep">column separator</param>
        /// <param name="nrows">number of rows to read</param>
        /// <returns>DataFrame</returns>
        public static DataFrame ReadView(IDataView view, int nrows = -1)
        {
            var df = new DataFrame();
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
                    var v = uint.Parse(val);
                    res = DetermineDataKind(nbline == 0, DataKind.U4, res);
                    continue;
                }
                catch (Exception /*e*/)
                {
                    if (string.IsNullOrEmpty(val))
                    {
                        res = DetermineDataKind(nbline == 0, DataKind.U4, res);
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

        /// <summary>
        /// Determines the more generic type with two types.
        /// </summary>
        static DataKind DetermineDataKind(bool first, DataKind suggested, DataKind previous)
        {
            if (first)
                return suggested;
            else
                return MaxKind(suggested, previous);
        }

        /// <summary>
        /// Determines the more generic type with two types.
        /// </summary>
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
            if (a == DataKind.U4 || b == DataKind.U4)
                return DataKind.U4;
            if (a == DataKind.I4 || b == DataKind.I4)
                return DataKind.I4;
            if (a == DataKind.BL || b == DataKind.BL)
                return DataKind.BL;
            return DataKind.TX;
        }

        /// <summary>
        /// Changes the values for an entire row.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="values"></param>
        public void FillValues(int row, string[] values)
        {
            _data.FillValues(row, values);
        }

        #endregion

        #region comparison

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public static bool operator ==(DataFrame df1, DataFrame df2)
        {
            return df1._data == df2._data;
        }

        /// <summary>
        /// Exact difference between two dataframes.
        /// </summary>
        public static bool operator !=(DataFrame df1, DataFrame df2)
        {
            return df1._data != df2._data;
        }

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public bool Equals(DataFrame df)
        {
            return _data.Equals(df._data);
        }

        /// <summary>
        /// Approximated comparison between two dataframes.
        /// It returns 0 if the difference is below the precision
        /// or the difference otherwise, Inf if shapes or schema are different.
        /// </summary>
        public double AlmostEquals(DataFrame df, double precision = 1e-6f, bool exc = false, bool printDf = false)
        {
            if (exc && printDf)
            {
                try
                {
                    return _data.AlmostEquals(df._data, precision, exc);
                }
                catch (Exception e)
                {
                    var addition = $"----\n{ToString()}\n-----\n{df.ToString()}";
                    throw new Exception(addition, e);
                }
            }
            else
                return _data.AlmostEquals(df._data, precision, exc);
        }

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public bool Equals(IDataFrameView dfv)
        {
            return Equals(dfv.Copy());
        }

        /// <summary>
        /// Exact comparison between two dataframes.
        /// </summary>
        public override bool Equals(object o)
        {
            var df = o as DataFrame;
            if (df == null)
                return false;
            return Equals(df);
        }

        /// <summary>
        /// Not implemented.
        /// </summary>
        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

        #endregion

        #region EntryPoints

        public Data.TextLoader EPTextLoader(string dataPath, char sep = ',', bool header = true)
        {
            var loader = new Data.TextLoader(dataPath)
            {
                Arguments = new Data.TextLoaderArguments()
                {
                    Separator = new[] { sep },
                    HasHeader = header,
                    Column = SchemaHelper.ToColumnArgArray(Schema)
                }
            };
            return loader;
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
            DataFrame _parent;

            public Iloc(DataFrame parent)
            {
                _parent = parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, int col]
            {
                get { return _parent._data[row, col]; }
                set { _parent._data[row, col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, int col]
            {
                set { _parent._data[rows, col] = value; }
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
            DataFrame _parent;

            public Loc(DataFrame parent)
            {
                _parent = parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, string col]
            {
                get { return _parent._data[row, col]; }
                set { _parent._data[row, col] = value; }
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[string col]
            {
                set { _parent._data[col].Set(value); }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, string col]
            {
                set { _parent._data[rows, col] = value; }
            }
        }

        #endregion

        #region operators []

        /// <summary>
        /// Returns all values in a row as a dictionary.
        /// </summary>
        public Dictionary<string, object> this[int row]
        {
            get { return _data[row]; }
        }

        /// <summary>
        /// Retrieves a column by its name.
        /// </summary>
        public NumericColumn GetColumn(string colname, int[] rows = null)
        {
            return new NumericColumn(_data.GetColumn(colname, rows));
        }

        /// <summary>
        /// Retrieves a column by its position.
        /// </summary>
        public NumericColumn GetColumn(int col, int[] rows = null)
        {
            return new NumericColumn(_data.GetColumn(col, rows));
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public NumericColumn this[string colname]
        {
            get { return GetColumn(colname); }
            set { AddColumn(colname, value as NumericColumn); }
        }

        /// <summary>
        /// Returns a subset of rows.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), null); }
        }

        /// <summary>
        /// Returns a subset of rows.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows]
        {
            get { return new DataFrameView(this, rows, null); }
        }

        /// <summary>
        /// Returns a subset of rows.
        /// </summary>
        public DataFrameView this[NumericColumn boolCol]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(boolCol), null); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, int colname]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), new[] { colname }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, IEnumerable<int> colnames]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), colnames); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, int colname]
        {
            get { return new DataFrameView(this, rows, new[] { colname }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, IEnumerable<int> colnames]
        {
            get { return new DataFrameView(this, rows, colnames); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, string colname]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), new[] { _data.GetColumnIndex(colname) }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<bool> rows, IEnumerable<string> colnames]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), colnames.Select(c => _data.GetColumnIndex(c))); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, string colname]
        {
            get { return new DataFrameView(this, rows, new[] { _data.GetColumnIndex(colname) }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[IEnumerable<int> rows, IEnumerable<string> colnames]
        {
            get { return new DataFrameView(this, rows, colnames.Select(c => _data.GetColumnIndex(c))); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[NumericColumn rows, string colname]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), new[] { _data.GetColumnIndex(colname) }); }
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public DataFrameView this[NumericColumn rows, IEnumerable<string> colnames]
        {
            get { return new DataFrameView(this, _data.EnumerateRowsIndex(rows), colnames.Select(c => _data.GetColumnIndex(c))); }
        }

        #endregion
    }
}
