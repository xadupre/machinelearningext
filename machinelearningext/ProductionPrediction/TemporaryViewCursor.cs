// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Creates a view on one row.
    /// </summary>
    /// <typeparam name="TRepValue"></typeparam>
    public class TemporaryViewCursor<TRepValue> : IDataView
    {
        readonly int _column;
        readonly TRepValue _value;
        readonly ISchema _schema;
        readonly IRowCursor _otherValues;

        public TRepValue Constant { get { return _value; } }
        public int ConstantCol { get { return _column; } }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="value">value which will replace the value</param>
        /// <param name="column">column to be replaced</param>
        /// <param name="otherValues">cursor which contains the others values</param>
        /// <param name="schema">schema to replace if otherValues is null</param>
        public TemporaryViewCursor(TRepValue value, int column, IRowCursor otherValues, ISchema schema = null)
        {
            _column = column;
            _value = value;
            _otherValues = otherValues;
            _schema = otherValues == null ? schema : otherValues.Schema;
            Contracts.AssertValue(_schema);
        }

        public bool CanShuffle { get { return false; } }
        public long? GetRowCount(bool lazy = true) { return null; }
        public ISchema Schema { get { return _schema; } }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return new CursorType(this, needCol, _otherValues);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            consolidator = new Consolidator();
            return new IRowCursor[] { GetRowCursor(needCol, rand) };
        }

        class Consolidator : IRowCursorConsolidator
        {
            public IRowCursor CreateCursor(IChannelProvider provider, IRowCursor[] inputs)
            {
                Contracts.Assert(inputs.Length == 1);
                return inputs[0];
            }
        }

        class CursorType : IRowCursor
        {
            Func<int, bool> _needCol;
            TemporaryViewCursor<TRepValue> _view;
            CursorState _state;
            IRowCursor _otherValues;

            public CursorType(TemporaryViewCursor<TRepValue> view, Func<int, bool> needCol, IRowCursor otherValues)
            {
                _needCol = needCol;
                _view = view;
                _state = CursorState.NotStarted;
                _otherValues = otherValues;
            }

            public CursorState State { get { return _state; } }
            public ICursor GetRootCursor() { return this; }
            public long Batch { get { return 1; } }
            public long Position { get { return 0; } }
            public ISchema Schema { get { return _view.Schema; } }
            public ValueGetter<UInt128> GetIdGetter() { return (ref UInt128 uid) => { uid = new UInt128(0, 1); }; }

            void IDisposable.Dispose()
            {
                if (_otherValues != null)
                {
                    // Do not dispose the cursor. The current one does not call MoveNext,
                    // it does not own any cursor and should free any of them.
                    // _otherValues.Dispose();
                    _otherValues = null;
                }
                GC.SuppressFinalize(this);
            }

            public bool MoveMany(long count)
            {
                throw Contracts.ExceptNotSupp();
            }

            public bool MoveNext()
            {
                if (State == CursorState.NotStarted)
                {
                    _state = CursorState.Good;
                    return true;
                }
                else if (State == CursorState.Good)
                {
                    _state = CursorState.Done;
                    return false;
                }
                return false;
            }

            public bool IsColumnActive(int col)
            {
                return col == _view._column || (_otherValues != null && _otherValues.IsColumnActive(col));
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (col == _view.ConstantCol)
                    return GetGetterPrivate(col) as ValueGetter<TValue>;
                else if (_otherValues != null)
                    return _otherValues.GetGetter<TValue>(col);
                else
                    throw Contracts.Except("otherValues is null, unable to access other columns.");
            }

            public ValueGetter<VBuffer<TRepValueItem>> GetGetterPrivateVector<TRepValueItem>(int col)
            {
                if (col == _view.ConstantCol)
                {
                    VBuffer<TRepValueItem> cast = (VBuffer<TRepValueItem>)(object)_view.Constant;
                    return (ref VBuffer<TRepValueItem> value) =>
                    {
                        cast.CopyTo(ref value);
                    };
                }
                else
                    throw Contracts.ExceptNotSupp("Unable to create a vector getter.");
            }

            public ValueGetter<TRepValue> GetGetterPrivate(int col)
            {
                if (col == _view.ConstantCol)
                {
                    var type = _view.Schema.GetColumnType(col);
                    if (type.IsVector)
                    {
                        switch (type.AsVector.ItemType.RawKind)
                        {
                            case DataKind.R4:
                                return GetGetterPrivateVector<float>(col) as ValueGetter<TRepValue>;
                            default:
                                throw Contracts.ExceptNotSupp("Unable to get a getter for type {0}", type.ToString());
                        }
                    }
                    else
                    {
                        return (ref TRepValue value) =>
                        {
                            value = _view.Constant;
                        };
                    }
                }
                else
                    throw Contracts.ExceptNotSupp();
            }
        }
    }
}
