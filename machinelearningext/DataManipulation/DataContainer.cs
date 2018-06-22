// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Ext.PipelineHelper;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Raised when there is a type mismatch.
    /// </summary>
    public class DataTypeError : Exception
    {
        public DataTypeError(string msg) : base(msg)
        {
        }
    }

    /// <summary>
    /// Implements a dense column container.
    /// </summary>
    public class DataColumn<DType> : IDataColumn, IEquatable<DataColumn<DType>>
        where DType : IEquatable<DType>
    {
        /// <summary>
        /// Data for the column.
        /// </summary>
        DType[] _data;

        /// <summary>
        /// Number of elements.
        /// </summary>
        public int Length => (_data == null ? 0 : _data.Length);

        public object Get(int row) { return _data[row]; }
        public void Set(int row, object value) { Set(row, (DType)value); }

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
        /// Creates a getter on the column. The getter returns the element at
        /// cursor.Position.
        /// </summary>
        public ValueGetter<DType> GetGetter(IRowCursor cursor)
        {
            return (ref DType value) => { value = _data[cursor.Position]; };
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
    }

    /// <summary>
    /// Contains data.
    /// </summary>
    public class DataContainer : IEquatable<DataContainer>
    {
        #region All possible types to hold data.

        IHost _host;
        List<string> _names;
        List<DataKind> _kinds;
        int _length;
        Dictionary<string, int> _naming;
        Dictionary<int, Tuple<DataKind, int>> _mapping;

        List<DataColumn<DvBool>> _colsBL;
        List<DataColumn<DvInt4>> _colsI4;
        List<DataColumn<DvInt8>> _colsI8;
        List<DataColumn<float>> _colsR4;
        List<DataColumn<double>> _colsR8;
        List<DataColumn<DvText>> _colsTX;

        #endregion

        /// <summary>
        /// Data Container.
        /// </summary>
        public DataContainer(IHost host)
        {
            _host = host;
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
            _colsI4 = null;
            _colsI8 = null;
            _colsR4 = null;
            _colsR8 = null;
            _colsTX = null;
            _mapping = new Dictionary<int, Tuple<DataKind, int>>();
            _naming = new Dictionary<string, int>();
        }

        /// <summary>
        /// Returns the dimension of the container.
        /// </summary>
        public Tuple<int, int> Shape => new Tuple<int, int>(Length, _names.Count);

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
        /// Returns the name of a column.
        /// </summary>
        public int GetColumnIndex(string name) { return _naming[name]; }

        /// <summary>
        /// Returns the type of a column.
        /// </summary>
        public DataKind GetDType(int col) { return _kinds[col]; }

        /// <summary>
        /// Returns the container of column col.
        /// </summary>
        public void GetTypedColumn<DType>(int col, out DataColumn<DType> column)
            where DType : IEquatable<DType>
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
                throw new DataTypeError(string.Format("Column {0} is not of type {1}", col, typeof(DType)));
            column = found;
        }

        /// <summary>
        /// Returns the container of column col.
        /// </summary>
        public IDataColumn GetColumn(int col)
        {
            var coor = _mapping[col];
            switch (coor.Item1)
            {
                case DataKind.BL:
                    DataColumn<DvBool> objbl;
                    GetTypedColumn(col, out objbl);
                    return objbl;
                case DataKind.I4:
                    DataColumn<DvInt4> obji4;
                    GetTypedColumn(col, out obji4);
                    return obji4;
                case DataKind.I8:
                    DataColumn<DvInt8> obji8;
                    GetTypedColumn(col, out obji8);
                    return obji8;
                case DataKind.R4:
                    DataColumn<float> objf;
                    GetTypedColumn(col, out objf);
                    return objf;
                case DataKind.R8:
                    DataColumn<double> objd;
                    GetTypedColumn(col, out objd);
                    return objd;
                case DataKind.TX:
                    DataColumn<DvText> objs;
                    GetTypedColumn(col, out objs);
                    return objs;
                default:
                    throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
            }
        }

        /// <summary>
        /// Adds a new column.
        /// </summary>
        /// <param name="name">column name</param>
        /// <param name="kind">column type</param>
        /// <param name="length">changes the length</param>
        public int AddColumn(string name, DataKind kind, int? length)
        {
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
                        _colsBL = new List<DataColumn<DvBool>>();
                    pos = _colsBL.Count;
                    _colsBL.Add(new DataColumn<DvBool>(nb));
                    break;
                case DataKind.I4:
                    if (_colsI4 == null)
                        _colsI4 = new List<DataColumn<DvInt4>>();
                    pos = _colsI4.Count;
                    _colsI4.Add(new DataColumn<DvInt4>(nb));
                    break;
                case DataKind.I8:
                    if (_colsI8 == null)
                        _colsI8 = new List<DataColumn<DvInt8>>();
                    pos = _colsI8.Count;
                    _colsI8.Add(new DataColumn<DvInt8>(nb));
                    break;
                case DataKind.R4:
                    if (_colsR4 == null)
                        _colsR4 = new List<DataColumn<float>>();
                    pos = _colsR4.Count;
                    _colsR4.Add(new DataColumn<float>(nb));
                    break;
                case DataKind.R8:
                    if (_colsR8 == null)
                        _colsR8 = new List<DataColumn<double>>();
                    pos = _colsR8.Count;
                    _colsR8.Add(new DataColumn<double>(nb));
                    break;
                case DataKind.TX:
                    if (_colsTX == null)
                        _colsTX = new List<DataColumn<DvText>>();
                    pos = _colsTX.Count;
                    _colsTX.Add(new DataColumn<DvText>(nb));
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
        /// Changes a value.
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
                    _colsBL[coor.Item2].Set(row, int.Parse(value));
                    break;
                case DataKind.I4:
                    _colsI4[coor.Item2].Set(row, int.Parse(value));
                    break;
                case DataKind.I8:
                    _colsI8[coor.Item2].Set(row, Int64.Parse(value));
                    break;
                case DataKind.R4:
                    _colsR4[coor.Item2].Set(row, float.Parse(value));
                    break;
                case DataKind.R8:
                    _colsR8[coor.Item2].Set(row, double.Parse(value));
                    break;
                case DataKind.TX:
                    _colsTX[coor.Item2].Set(row, value);
                    break;
                default:
                    throw new DataTypeError(string.Format("Type {0} is not handled.", coor.Item1));
            }
        }

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

        public ISchema Schema => new DataContainerSchema(this);

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return new RowCursor(this, needCol, rand);
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

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            n = DataViewUtils.GetThreadCount(_host, n);
            if (n > 1 && (long)n > Length)
                n = (int)Length;

            if (n <= 1)
            {
                consolidator = null;
                return new IRowCursor[] { GetRowCursor(predicate, rand) };
            }
            else
            {
                consolidator = new Consolidator();
                var cursors = new IRowCursor[n];
                for (int i = 0; i < cursors.Length; ++i)
                    cursors[i] = new RowCursor(this, predicate, rand, n, i);
                return cursors;
            }
        }

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

        void FillValues(IRowCursor cursor, Dictionary<int, Tuple<int, int>> memory)
        {
            var getterBL = new ValueGetter<DvBool>[_colsBL == null ? 0 : _colsBL.Count];
            var getterI4 = new ValueGetter<DvInt4>[_colsI4 == null ? 0 : _colsI4.Count];
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

        public class RowCursor : IRowCursor
        {
            DataContainer _cont;
            public long Position => _position;
            public long Batch => _batch;
            IRandom _rand;
            Func<int, bool> _needCol;
            long _inc;
            long _first;
            long _position;
            long _batch;

            public RowCursor(DataContainer cont, Func<int, bool> needCol, IRandom rand = null, int inc = 1, int first = 0)
            {
                _cont = cont;
                _position = -1;
                _batch = -1;
                _inc = inc;
                _first = first;
                _needCol = needCol;
                _rand = rand;
                if (rand != null)
                    throw new NotImplementedException();
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
            public ISchema Schema => _cont.Schema;

            public CursorState State
            {
                get
                {
                    if (Position == -1)
                        return CursorState.NotStarted;
                    if (Position < _cont.Length)
                        return CursorState.Good;
                    return CursorState.Done;
                }
            }

            public bool MoveMany(long count)
            {
                if (_position == -1)
                    _position = _inc * (count - 1) + _first;
                else
                    _position += count * _inc;
                return _position < _cont.Length;
            }

            public bool MoveNext()
            {
                if (_position == -1)
                    _position = _first;
                else
                    _position += _inc;
                return _position < _cont.Length;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                var coor = _cont._mapping[col];
                switch (coor.Item1)
                {
                    case DataKind.BL:
                        return _cont._colsBL[coor.Item2].GetGetter(this) as ValueGetter<TValue>;
                    case DataKind.I4:
                        return _cont._colsI4[coor.Item2].GetGetter(this) as ValueGetter<TValue>;
                    case DataKind.I8:
                        return _cont._colsI8[coor.Item2].GetGetter(this) as ValueGetter<TValue>;
                    case DataKind.R4:
                        return _cont._colsR4[coor.Item2].GetGetter(this) as ValueGetter<TValue>;
                    case DataKind.R8:
                        return _cont._colsR8[coor.Item2].GetGetter(this) as ValueGetter<TValue>;
                    case DataKind.TX:
                        return _cont._colsTX[coor.Item2].GetGetter(this) as ValueGetter<TValue>;
                    default:
                        throw new NotImplementedException();
                }
            }
        }

        #endregion

        #region pandas API, operators (slow)

        public object this[int row, int col]
        {
            get
            {
                return GetColumn(col).Get(row);
            }
            set
            {
                GetColumn(col).Set(row, value);
            }
        }

        public static bool operator ==(DataContainer c1, DataContainer c2)
        {
            return c1.Equals(c2);
        }

        public static bool operator !=(DataContainer c1, DataContainer c2)
        {
            return !c1.Equals(c2);
        }

        public override bool Equals(object c)
        {
            var cd = c as DataContainer;
            if (cd == null)
                return false;
            return Equals(cd);
        }

        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }

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
            if (_colsI4 != null)
            {
                for (int i = 0; i < _colsI4.Count; ++i)
                    if (!_colsI4[i].Equals(c._colsI4[i]))
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
    }
}
