// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Scikit.ML.ProductionPrediction
{
    /// <summary>
    /// Creates a view on one row and can loop on it after it was replaced.
    /// </summary>
    /// <typeparam name="TRepValue"></typeparam>
    public class InfiniteLoopViewCursor<TRepValue> : IDataView
    {
        readonly int _column;
        readonly ISchema _schema;
        readonly IRowCursor _otherValues;
        CursorType _ownCursor;

        public int ConstantCol { get { return _column; } }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="column">column to be replaced</param>
        /// <param name="otherValues">cursor which contains the others values</param>
        /// <param name="schema">schema of the </param>
        public InfiniteLoopViewCursor(int column, IRowCursor otherValues, ISchema schema = null)
        {
            _column = column;
            _otherValues = otherValues;
            _schema = otherValues == null ? schema : otherValues.Schema;
            _ownCursor = null;
            Contracts.AssertValue(_schema);
        }

        public bool CanShuffle { get { return false; } }
        public long? GetRowCount(bool lazy = true) { return null; }
        public ISchema Schema { get { return _schema; } }

        public void Set(ref TRepValue value)
        {
            if (_ownCursor == null)
                throw Contracts.Except("GetRowCursor on this view was never called. No cursor is registered.");
            _ownCursor.Set(ref value);
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            if (_ownCursor != null)
                throw Contracts.Except("GetRowCursor was called a second time which probably means this function was called from multiple threads. " +
                    "Be sure that TlcEnvironment is called by parameter conc:1.");
            _ownCursor = new CursorType(this, needCol, _otherValues);
            return _ownCursor;
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            var cur = GetRowCursor(needCol, rand);
            consolidator = null;
            return new IRowCursor[] { cur };
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
            InfiniteLoopViewCursor<TRepValue> _view;
            CursorState _state;
            IRowCursor _otherValues;
            TRepValue[] _container;
            bool _wait;
            long _position;
            long _batch;

            public CursorType(InfiniteLoopViewCursor<TRepValue> view, Func<int, bool> needCol, IRowCursor otherValues)
            {
                _needCol = needCol;
                _view = view;
                _state = CursorState.NotStarted;
                _container = new TRepValue[1];
                _otherValues = otherValues;
                _wait = true;
                _position = 0;
                _batch = 1;
            }

            public CursorState State { get { return _state; } }
            public ICursor GetRootCursor() { return this; }
            public long Batch { get { return _batch; } }
            public long Position { get { return _position; } }
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
                if (State == CursorState.Done)
                    throw Contracts.Except("The state of the cursor should not be Done.");
                if (_wait)
                    throw Contracts.Except("The cursor has no value to show. This exception happens because a different thread is " +
                        "requested the next value or because a view is requesting for more than one value at a time.");
                _state = CursorState.Good;
                ++_position;
                ++_batch;
                _wait = false;
                return true;
            }

            public void Set(ref TRepValue value)
            {
                _container[0] = value;
                _wait = false;
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
                    throw Contracts.Except("otherValues is null, unable to access other columns." +
                        "If you are using PrePostTransformPredictor, it means the preprossing transform cannot be " +
                        "converted into a IValueMapper because it relies on more than one columns.");
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
                            value = _container[0];
                        };
                    }
                }
                else
                    throw Contracts.ExceptNotSupp();
            }

            public ValueGetter<VBuffer<TRepValueItem>> GetGetterPrivateVector<TRepValueItem>(int col)
            {
                if (col == _view.ConstantCol)
                {
                    return (ref VBuffer<TRepValueItem> value) =>
                    {
                        VBuffer<TRepValueItem> cast = (VBuffer<TRepValueItem>)(object)_container[0];
                        cast.CopyTo(ref value);
                    };
                }
                else
                    throw Contracts.ExceptNotSupp("Unable to create a vector getter.");
            }
        }
    }
}
