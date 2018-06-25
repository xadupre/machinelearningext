// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Contains data.
    /// </summary>
    public class DataContainer : IEquatable<DataContainer>
    {
        #region All possible types to hold data.

        List<string> _names;
        List<DataKind> _kinds;
        int _length;
        Dictionary<string, int> _naming;
        Dictionary<int, Tuple<DataKind, int>> _mapping;
        ISchema _schema;

        List<IDataColumn> _colsBL;
        List<IDataColumn> _colsI4;
        List<IDataColumn> _colsU4;
        List<IDataColumn> _colsI8;
        List<IDataColumn> _colsR4;
        List<IDataColumn> _colsR8;
        List<IDataColumn> _colsTX;

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public DataContainer Copy()
        {
            var dc = new DataContainer();
            dc._names = new List<string>(_names);
            dc._kinds = new List<DataKind>(_kinds);
            dc._length = _length;
            dc._naming = new Dictionary<string, int>(_naming);
            dc._mapping = _mapping = new Dictionary<int, Tuple<DataKind, int>>(_mapping);
            dc._schema = new DataContainerSchema(dc);
            dc._colsBL = _colsBL == null ? null : new List<IDataColumn>(_colsBL.Select(c => c.Copy()));
            dc._colsI4 = _colsI4 == null ? null : new List<IDataColumn>(_colsI4.Select(c => c.Copy()));
            dc._colsU4 = _colsU4 == null ? null : new List<IDataColumn>(_colsU4.Select(c => c.Copy()));
            dc._colsI8 = _colsI8 == null ? null : new List<IDataColumn>(_colsI8.Select(c => c.Copy()));
            dc._colsR4 = _colsR4 == null ? null : new List<IDataColumn>(_colsR4.Select(c => c.Copy()));
            dc._colsR8 = _colsR8 == null ? null : new List<IDataColumn>(_colsR8.Select(c => c.Copy()));
            dc._colsTX = _colsTX == null ? null : new List<IDataColumn>(_colsTX.Select(c => c.Copy()));
            return dc;
        }

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public DataContainer Copy(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            var arows = rows.ToArray();
            var dc = new DataContainer();
            foreach (var c in columns)
                dc.AddColumn(_names[c], _kinds[c], arows.Length, GetColumn(c).Copy(arows));
            return dc;
        }

        /// <summary>
        /// Converts a filter into a list of row indices.
        /// </summary>
        public IEnumerable<int> EnumerateRowsIndex(IEnumerable<bool> filter)
        {
            int row = 0;
            foreach (var b in filter)
            {
                if (b)
                    yield return row;
                ++row;
            }
        }

        /// <summary>
        /// Returns the dimension of the container.
        /// </summary>
        public Tuple<int, int> Shape => new Tuple<int, int>(Length, _names.Count);

        /// <summary>
        /// Returns the name and the type of a column such as
        /// <pre>name:type:index</pre>.
        /// </summary>
        public string NameType(int col) { return string.Format("{0}:{1}:{2}", _names[col], _kinds[col], col); }

        /// <summary>
        /// Returns the number of rows.
        /// </summary>
        public int Length => _length;

        /// <summary>
        /// Returns the name of a column.
        /// </summary>
        public string GetColumnName(int col) { return _names[col]; }

        /// <summary>
        /// Returns the position of a column.
        /// </summary>
        public int GetColumnIndex(string name) { return _naming[name]; }

        /// <summary>
        /// Returns the type of a column.
        /// </summary>
        public DataKind GetDType(int col) { return _kinds[col]; }

        /// <summary>
        /// Returns a typed container of column col.
        /// </summary>
        public void GetTypedColumn<DType>(int col, out DataColumn<DType> column, int[] rows = null)
            where DType : IEquatable<DType>, IComparable<DType>
        {
            var coor = _mapping[col];
            DataColumn<DType> found = null;
            switch (coor.Item1)
            {
                case DataKind.BL:
                    found = _colsBL[coor.Item2] as DataColumn<DType>;
                    break;
                case DataKind.I4:
                    found = _colsI4[coor.Item2] as DataColumn<DType>;
                    break;
                case DataKind.U4:
                    found = _colsU4[coor.Item2] as DataColumn<DType>;
                    break;
                case DataKind.I8:
                    found = _colsI8[coor.Item2] as DataColumn<DType>;
                    break;
                case DataKind.R4:
                    found = _colsR4[coor.Item2] as DataColumn<DType>;
                    break;
                case DataKind.R8:
                    found = _colsR8[coor.Item2] as DataColumn<DType>;
                    break;
                case DataKind.TX:
                    found = _colsTX[coor.Item2] as DataColumn<DType>;
                    break;
                default:
                    throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
            }
            if (found == null)
                throw new DataTypeError(string.Format("Column {0} is not of type {1} (kind={2}, c={3})",
                                col, typeof(DType), coor.Item1, coor.Item2));
            if (rows != null)
            {
                // TODO: This is not absolutely efficient as it copies the data.
                found = found.Copy(rows) as DataColumn<DType>;
            }
            column = found;
        }

        /// <summary>
        /// Returns the container of column col as an interface.
        /// </summary>
        public IDataColumn GetColumn(string colname, int[] rows = null)
        {
            return GetColumn(_naming[colname], rows);
        }

        /// <summary>
        /// Returns the container of column col as an interface.
        /// </summary>
        public IDataColumn GetColumn(int col, int[] rows = null)
        {
            var coor = _mapping[col];
            switch (coor.Item1)
            {
                case DataKind.BL:
                    DataColumn<DvBool> objbl;
                    GetTypedColumn(col, out objbl, rows);
                    return objbl;
                case DataKind.I4:
                    DataColumn<DvInt4> obji4;
                    GetTypedColumn(col, out obji4, rows);
                    return obji4;
                case DataKind.U4:
                    DataColumn<uint> obju4;
                    GetTypedColumn(col, out obju4, rows);
                    return obju4;
                case DataKind.I8:
                    DataColumn<DvInt8> obji8;
                    GetTypedColumn(col, out obji8, rows);
                    return obji8;
                case DataKind.R4:
                    DataColumn<float> objf;
                    GetTypedColumn(col, out objf, rows);
                    return objf;
                case DataKind.R8:
                    DataColumn<double> objd;
                    GetTypedColumn(col, out objd, rows);
                    return objd;
                case DataKind.TX:
                    DataColumn<DvText> objs;
                    GetTypedColumn(col, out objs, rows);
                    return objs;
                default:
                    throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
            }
        }

        #endregion

        #region constructor

        /// <summary>
        /// Data Container.
        /// </summary>
        public DataContainer()
        {
            _init();
        }

        /// <summary>
        /// Initializes the class as empty.
        /// </summary>
        void _init()
        {
            _length = 0;
            _names = null;
            _kinds = null;
            _colsBL = null;
            _colsI4 = null;
            _colsU4 = null;
            _colsI8 = null;
            _colsR4 = null;
            _colsR8 = null;
            _colsTX = null;
            _mapping = new Dictionary<int, Tuple<DataKind, int>>();
            _naming = new Dictionary<string, int>();
            _schema = new DataContainerSchema(this);
        }

        #endregion

        #region column

        /// <summary>
        /// Adds a new column.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="kind">column type</param>
        /// <param name="length">changes the length</param>
        /// <param name="values">values (can be null)</param>
        public int AddColumn(string name, DataKind kind, int? length, IDataColumn values = null)
        {
            if (_naming.ContainsKey(name))
                throw new DataNameError(string.Format("Column '{0}' already exists, it cannot be created again.", name));
            if (values != null && (values as NumericColumn != null))
                values = (values as NumericColumn).Column;
            if (_names == null)
                _names = new List<string>();
            if (_kinds == null)
                _kinds = new List<DataKind>();
            if (length.HasValue && Length > 0 && length.Value != Length)
                throw new DataTypeError(string.Format("Length mismatch, expected {0} got {1}", Length, length.Value));

            int last = _names.Count;
            _names.Add(name);
            _kinds.Add(kind);
            int pos = -1;
            int nb = length ?? Length;
            _length = nb;
            switch (kind)
            {
                case DataKind.BL:
                    if (_colsBL == null)
                        _colsBL = new List<IDataColumn>();
                    pos = _colsBL.Count;
                    _colsBL.Add(values ?? new DataColumn<DvBool>(nb));
                    break;
                case DataKind.I4:
                    if (_colsI4 == null)
                        _colsI4 = new List<IDataColumn>();
                    pos = _colsI4.Count;
                    _colsI4.Add(values ?? new DataColumn<DvInt4>(nb));
                    break;
                case DataKind.U4:
                    if (_colsU4 == null)
                        _colsU4 = new List<IDataColumn>();
                    pos = _colsU4.Count;
                    _colsU4.Add(values ?? new DataColumn<uint>(nb));
                    break;
                case DataKind.I8:
                    if (_colsI8 == null)
                        _colsI8 = new List<IDataColumn>();
                    pos = _colsI8.Count;
                    _colsI8.Add(values ?? new DataColumn<DvInt8>(nb));
                    break;
                case DataKind.R4:
                    if (_colsR4 == null)
                        _colsR4 = new List<IDataColumn>();
                    pos = _colsR4.Count;
                    _colsR4.Add(values ?? new DataColumn<float>(nb));
                    break;
                case DataKind.R8:
                    if (_colsR8 == null)
                        _colsR8 = new List<IDataColumn>();
                    pos = _colsR8.Count;
                    _colsR8.Add(values ?? new DataColumn<double>(nb));
                    break;
                case DataKind.TX:
                    if (_colsTX == null)
                        _colsTX = new List<IDataColumn>();
                    pos = _colsTX.Count;
                    _colsTX.Add(values ?? new DataColumn<DvText>(nb));
                    break;
                default:
                    throw new DataTypeError(string.Format("Type {0} is not handled.", kind));
            }
            _mapping[last] = new Tuple<DataKind, int>(kind, pos);
            _naming[name] = last;
            return last;
        }

        /// <summary>
        /// Sets values for a row.
        /// </summary>
        /// <param name="row">row to fill</param>
        /// <param name="spl">value as strings, they will be converted</param>
        public void FillValues(int row, string[] spl)
        {
            for (int i = 0; i < spl.Length; ++i)
            {
                Set(row, i, spl[i]);
            }
        }

        /// <summary>
        /// Changes a value at a specific location.
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="col">column</param>
        /// <param name="value">value</param>
        public void Set(int row, int col, string value)
        {
            var coor = _mapping[col];
            switch (_kinds[col])
            {
                case DataKind.BL:
                    _colsBL[coor.Item2].Set(row, (DvBool)bool.Parse(value));
                    break;
                case DataKind.I4:
                    _colsI4[coor.Item2].Set(row, (DvInt4)int.Parse(value));
                    break;
                case DataKind.U4:
                    _colsU4[coor.Item2].Set(row, uint.Parse(value));
                    break;
                case DataKind.I8:
                    _colsI8[coor.Item2].Set(row, (DvInt8)Int64.Parse(value));
                    break;
                case DataKind.R4:
                    _colsR4[coor.Item2].Set(row, float.Parse(value));
                    break;
                case DataKind.R8:
                    _colsR8[coor.Item2].Set(row, double.Parse(value));
                    break;
                case DataKind.TX:
                    _colsTX[coor.Item2].Set(row, new DvText(value));
                    break;
                default:
                    throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
            }
        }

        #endregion

        #region IDataView API

        /// <summary>
        /// Implements ISchema interface for this container.
        /// </summary>
        public class DataContainerSchema : ISchema
        {
            DataContainer _cont;
            public DataContainerSchema(DataContainer cont) { _cont = cont; }
            public int ColumnCount => _cont._names.Count;
            public string GetColumnName(int col) { return _cont._names[col]; }
            public bool TryGetColumnIndex(string name, out int col)
            {
                return _cont._naming.TryGetValue(name, out col);
            }
            public ColumnType GetColumnType(int col) { return SchemaHelper.DataKind2ColumnType(_cont._kinds[col]); }
            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                if (kind == _cont._kinds[col].ToString())
                    return GetColumnType(col);
                return null;
            }
            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                throw new NotImplementedException();
            }
            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Returns the schema.
        /// </summary>
        public ISchema Schema => _schema;

        /// <summary>
        /// Fills the value with values coming from a IDataView.
        /// nrow must be specified for the first column.
        /// The method checks that all column have the same number of elements.
        /// </summary>
        public void FillValues(IDataView view, int nrows = -1)
        {
            long? numRows = view.GetRowCount(false);
            if (!numRows.HasValue)
                numRows = DataViewUtils.ComputeRowCount(view);

            var memory = new Dictionary<int, Tuple<int, int>>();
            var sch = view.Schema;
            int pos = 0;
            for (int i = 0; i < sch.ColumnCount; ++i)
            {
                if (sch.IsHidden(i))
                    continue;
                var ty = sch.GetColumnType(i);
                if (ty.IsVector)
                {
                    var tyv = ty.AsVector;
                    if (tyv.DimCount != 1)
                        throw new NotSupportedException("Only arrays with one dimension are supported.");
                    for (int j = 0; j < tyv.GetDim(0); ++j)
                    {
                        AddColumn(string.Format("{0}.{1}", sch.GetColumnName(i), j), tyv.ItemType.RawKind, (int)numRows.Value);
                        memory[pos++] = new Tuple<int, int>(i, j);
                    }
                }
                else
                {
                    memory[pos] = new Tuple<int, int>(i, -1);
                    AddColumn(sch.GetColumnName(i), ty.RawKind, (int)numRows.Value);
                    ++pos;
                }
            }

            // Fills values.
            using (var cursor = view.GetRowCursor(i => true))
                FillValues(cursor, memory);
        }

        /// <summary>
        /// Fills the value with values coming from a IRowCursor.
        /// Called by the previous method.
        /// </summary>
        void FillValues(IRowCursor cursor, Dictionary<int, Tuple<int, int>> memory)
        {
            var getterBL = new ValueGetter<DvBool>[_colsBL == null ? 0 : _colsBL.Count];
            var getterI4 = new ValueGetter<DvInt4>[_colsI4 == null ? 0 : _colsI4.Count];
            var getterU4 = new ValueGetter<uint>[_colsU4 == null ? 0 : _colsU4.Count];
            var getterI8 = new ValueGetter<DvInt8>[_colsI8 == null ? 0 : _colsI8.Count];
            var getterR4 = new ValueGetter<float>[_colsR4 == null ? 0 : _colsR4.Count];
            var getterR8 = new ValueGetter<double>[_colsR8 == null ? 0 : _colsR8.Count];
            var getterTX = new ValueGetter<DvText>[_colsTX == null ? 0 : _colsTX.Count];
            for (int i = 0; i < _names.Count; ++i)
            {
                var coor = _mapping[i];
                switch (coor.Item1)
                {
                    case DataKind.BL:
                        getterBL[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, DvBool.NA);
                        break;
                    case DataKind.I4:
                        getterI4[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, DvInt4.NA);
                        break;
                    case DataKind.U4:
                        getterU4[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, (uint)0);
                        break;
                    case DataKind.I8:
                        getterI8[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, DvInt8.NA);
                        break;
                    case DataKind.R4:
                        getterR4[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, float.NaN);
                        break;
                    case DataKind.R8:
                        getterR8[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, double.NaN);
                        break;
                    case DataKind.TX:
                        getterTX[coor.Item2] = GetGetterCursor(cursor, memory[i].Item1, memory[i].Item2, DvText.NA);
                        break;
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", coor.Item1));
                }
            }

            int row = 0;
            var valueBL = new DvBool();
            var valueI4 = new DvInt4();
            uint valueU4 = 0;
            var valueI8 = new DvInt8();
            float valueR4 = 0;
            double valueR8 = 0;
            var valueTX = new DvText();
            while (cursor.MoveNext())
            {
                for (int i = 0; i < _names.Count; ++i)
                {
                    var coor = _mapping[i];
                    switch (coor.Item1)
                    {
                        case DataKind.BL:
                            getterBL[coor.Item2](ref valueBL);
                            _colsBL[coor.Item2].Set(row, valueBL);
                            break;
                        case DataKind.I4:
                            getterI4[coor.Item2](ref valueI4);
                            _colsI4[coor.Item2].Set(row, valueI4);
                            break;
                        case DataKind.U4:
                            getterU4[coor.Item2](ref valueU4);
                            _colsU4[coor.Item2].Set(row, valueU4);
                            break;
                        case DataKind.I8:
                            getterI8[coor.Item2](ref valueI8);
                            _colsI8[coor.Item2].Set(row, valueI8);
                            break;
                        case DataKind.R4:
                            getterR4[coor.Item2](ref valueR4);
                            _colsR4[coor.Item2].Set(row, valueR4);
                            break;
                        case DataKind.R8:
                            getterR8[coor.Item2](ref valueR8);
                            _colsR8[coor.Item2].Set(row, valueR8);
                            break;
                        case DataKind.TX:
                            getterTX[coor.Item2](ref valueTX);
                            _colsTX[coor.Item2].Set(row, valueTX);
                            break;
                        default:
                            throw new NotImplementedException(string.Format("Not implemented for kind {0}", coor.Item1));
                    }
                }
                ++row;
            }
        }

        /// <summary>
        /// Returns a getter of a certain type.
        /// </summary>
        ValueGetter<DType> GetGetterCursor<DType>(IRowCursor cursor, int col, int index, DType defaultValue)
        {
            var dt = cursor.Schema.GetColumnType(col);
            if (dt.IsVector)
            {
                var getter = cursor.GetGetter<VBuffer<DType>>(col);
                var temp = new VBuffer<DType>();
                return (ref DType value) =>
                {
                    getter(ref temp);
                    if (temp.IsDense)
                        value = temp.Values[index];
                    else
                    {
                        value = defaultValue;
                        for (int j = 0; j < temp.Count; ++j)
                        {
                            if (temp.Indices[j] == index)
                            {
                                value = temp.Values[j];
                                break;
                            }
                        }
                    }
                };
            }
            else
                return cursor.GetGetter<DType>(col);
        }

        #endregion

        #region Cursor

        /// <summary>
        /// Returns a cursor on the data.
        /// </summary>
        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return new RowCursor(this, needCol, rand);
        }

        /// <summary>
        /// Returns a cursor on a subset of the data.
        /// </summary>
        public IRowCursor GetRowCursor(int[] rows, int[] columns, Func<int, bool> needCol, IRandom rand = null)
        {
            return new RowCursor(this, needCol, rand, rows: rows, columns: columns);
        }

        private sealed class Consolidator : IRowCursorConsolidator
        {
            private const int _batchShift = 6;
            private const int _batchSize = 1 << _batchShift;
            public IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs)
            {
                return DataViewUtils.ConsolidateGeneric(provider, inputs, _batchSize);
            }
        }

        /// <summary>
        /// Returns a set of aliased cursors on the data.
        /// </summary>
        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            return GetRowCursorSet(null, null, out consolidator, predicate, n, rand);
        }

        /// <summary>
        /// Returns a set of aliased cursors on the data.
        /// </summary>
        public IRowCursor[] GetRowCursorSet(int[] rows, int[] columns, out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            var host = new TlcEnvironment().Register("Estimate n threads");
            n = DataViewUtils.GetThreadCount(host, n);
            if (n > 1 && (long)n > Length)
                n = (int)Length;

            if (n <= 1)
            {
                consolidator = null;
                return new IRowCursor[] { GetRowCursor(rows, columns, predicate, rand) };
            }
            else
            {
                var cursors = new IRowCursor[n];
                for (int i = 0; i < cursors.Length; ++i)
                    cursors[i] = new RowCursor(this, predicate, rand, n, i, rows: rows, columns: columns);
                consolidator = new Consolidator();
                return cursors;
            }
        }

        /// <summary>
        /// Implements a cursor for this container.
        /// </summary>
        public class RowCursor : IRowCursor
        {
            DataContainer _cont;
            public long Batch => _first;
            IRandom _rand;
            Func<int, bool> _needCol;
            long _inc;
            long _first;
            long _position;

            int[] _rowsSet;
            int[] _colsSet;
            Dictionary<int, int> _revColsSet;
            ISchema _schema;

            public RowCursor(DataContainer cont, Func<int, bool> needCol,
                             IRandom rand = null, int inc = 1, int first = 0,
                             int[] rows = null, int[] columns = null)
            {
                _cont = cont;
                _position = -1;
                _inc = inc;
                _first = first;
                _needCol = needCol;
                _rand = rand;
                if (rand != null)
                    throw new NotImplementedException();
                _rowsSet = rows;
                _colsSet = columns;
                if (_colsSet != null)
                {
                    _revColsSet = new Dictionary<int, int>();
                    for (int i = 0; i < columns.Length; ++i)
                        _revColsSet[columns[i]] = i;
                    _schema = new DataFrameViewSchema(_cont.Schema, columns);
                }
                else
                    _schema = null;
            }

            public ValueGetter<UInt128> GetIdGetter()
            {
                return (ref UInt128 idrow) => { idrow = new UInt128(Position >= 0 ? (ulong)Position : (ulong)(-Position), Position >= 0 ? (ulong)1 : (ulong)0); };
            }

            public void Dispose()
            {
            }

            public ICursor GetRootCursor() { return this; }
            public bool IsColumnActive(int col) { return _needCol(col); }
            public ISchema Schema => _colsSet == null ? _cont.Schema : _schema;

            public long Position => _rowsSet == null ? _position : _rowsSet[_position];
            int LastPosition => _rowsSet == null ? _cont.Length : _rowsSet.Length;

            public CursorState State
            {
                get
                {
                    if (_position == -1)
                        return CursorState.NotStarted;
                    return _position < LastPosition ? CursorState.Good : CursorState.Done;
                }
            }

            public bool MoveMany(long count)
            {
                if (_position == -1)
                    _position = _inc * (count - 1) + _first;
                else
                    _position += count * _inc;
                return _position < LastPosition;
            }

            public bool MoveNext()
            {
                if (_position == -1)
                    _position = _first;
                else
                    _position += _inc;
                return _position < LastPosition;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                col = _colsSet == null ? col : _colsSet[col];
                var coor = _cont._mapping[col];
                switch (coor.Item1)
                {
                    case DataKind.BL:
                        return _cont._colsBL[coor.Item2].GetGetter<TValue>(this);
                    case DataKind.I4:
                        return _cont._colsI4[coor.Item2].GetGetter<TValue>(this);
                    case DataKind.U4:
                        return _cont._colsU4[coor.Item2].GetGetter<TValue>(this) as ValueGetter<TValue>;
                    case DataKind.I8:
                        return _cont._colsI8[coor.Item2].GetGetter<TValue>(this) as ValueGetter<TValue>;
                    case DataKind.R4:
                        return _cont._colsR4[coor.Item2].GetGetter<TValue>(this) as ValueGetter<TValue>;
                    case DataKind.R8:
                        return _cont._colsR8[coor.Item2].GetGetter<TValue>(this) as ValueGetter<TValue>;
                    case DataKind.TX:
                        return _cont._colsTX[coor.Item2].GetGetter<TValue>(this) as ValueGetter<TValue>;
                    default:
                        throw new NotImplementedException();
                }
            }
        }

        #endregion

        #region comparison

        /// <summary>
        /// Checks that containers are exactly the same.
        /// </summary>
        public static bool operator ==(DataContainer c1, DataContainer c2)
        {
            return c1.Equals(c2);
        }

        /// <summary>
        /// Checks that containers are different.
        /// </summary>
        public static bool operator !=(DataContainer c1, DataContainer c2)
        {
            return !c1.Equals(c2);
        }

        /// <summary>
        /// Checks that containers are exactly the same.
        /// </summary>
        public override bool Equals(object c)
        {
            var cd = c as DataContainer;
            if (cd == null)
                return false;
            return Equals(cd);
        }

        /// <summary>
        /// Not implemented.
        /// </summary>
        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Checks that containers are exactly the same.
        /// </summary>
        public bool Equals(DataContainer c)
        {
            if (Shape.Item1 != c.Shape.Item1 || Shape.Item2 != c.Shape.Item2)
                return false;
            for (int i = 0; i < _names.Count; ++i)
            {
                if (_names[i] != c._names[i])
                    return false;
                if (_kinds[i] != c._kinds[i])
                    return false;
                if (_mapping[i].Item1 != c._mapping[i].Item1 || _mapping[i].Item2 != c._mapping[i].Item2)
                    return false;
            }
            if (_colsBL != null)
            {
                for (int i = 0; i < _colsBL.Count; ++i)
                    if (!_colsBL[i].Equals(c._colsBL[i]))
                        return false;
            }
            if (_colsI4 != null)
            {
                for (int i = 0; i < _colsI4.Count; ++i)
                    if (!_colsI4[i].Equals(c._colsI4[i]))
                        return false;
            }
            if (_colsU4 != null)
            {
                for (int i = 0; i < _colsU4.Count; ++i)
                    if (!_colsU4[i].Equals(c._colsU4[i]))
                        return false;
            }
            if (_colsI8 != null)
            {
                for (int i = 0; i < _colsI8.Count; ++i)
                    if (!_colsI8[i].Equals(c._colsI8[i]))
                        return false;
            }
            if (_colsR4 != null)
            {
                for (int i = 0; i < _colsR4.Count; ++i)
                    if (!_colsR4[i].Equals(c._colsR4[i]))
                        return false;
            }
            if (_colsR8 != null)
            {
                for (int i = 0; i < _colsR8.Count; ++i)
                    if (!_colsR8[i].Equals(c._colsR8[i]))
                        return false;
            }
            if (_colsTX != null)
            {
                for (int i = 0; i < _colsTX.Count; ++i)
                    if (!_colsTX[i].Equals(c._colsTX[i]))
                        return false;
            }
            return true;
        }

        #endregion

        #region operators

        /// <summary>
        /// Usual operator [i,j].
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="col">column</param>
        /// <returns>value</returns>
        public object this[int row, int col]
        {
            get { return GetColumn(col).Get(row); }
            set { GetColumn(col).Set(row, value); }
        }

        /// <summary>
        /// Usual operator [i,colname].
        /// </summary>
        /// <param name="row">row</param>
        /// <param name="col">column</param>
        /// <returns>value</returns>
        public object this[int row, string colname]
        {
            get { return GetColumn(colname).Get(row); }
            set { GetColumn(colname).Set(row, value); }
        }


        /// <summary>
        /// Returns a column.
        /// </summary>
        public NumericColumn this[string colname]
        {
            get { return new NumericColumn(GetColumn(colname)); }
        }

        /// <summary>
        /// Returns all values in a row as a dictionary.
        /// </summary>
        public Dictionary<string, object> this[int row]
        {
            get
            {
                var res = new Dictionary<string, object>();
                for (int i = 0; i < _names.Count; ++i)
                    res[_names[i]] = this[row, i];
                return res;
            }
        }

        /// <summary>
        /// Changes the value of a column and a subset of rows.
        /// </summary>
        public object this[IEnumerable<bool> rows, int col]
        {
            set { GetColumn(col).Set(rows, value); }
        }

        /// <summary>
        /// Changes the value of a column and a subset of rows.
        /// </summary>
        public object this[IEnumerable<bool> rows, string col]
        {
            set { GetColumn(col).Set(rows, value); }
        }

        #endregion
    }
}
