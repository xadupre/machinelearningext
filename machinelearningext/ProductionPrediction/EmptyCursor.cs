// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.ProductionPrediction
{
    public class EmptyCursor : IRowCursor
    {
        Func<int, bool> _needCol;
        IDataView _view;
        CursorState _state;

        public EmptyCursor(IDataView view, Func<int, bool> needCol)
        {
            _needCol = needCol;
            _view = view;
            _state = CursorState.NotStarted;
        }

        public CursorState State { get { return _state; } }
        public ICursor GetRootCursor() { return this; }
        public long Batch { get { return 0; } }
        public long Position { get { return 0; } }
        public Schema Schema { get { return _view.Schema; } }
        public ValueGetter<UInt128> GetIdGetter() { return (ref UInt128 uid) => { uid = new UInt128(0, 1); }; }

        void IDisposable.Dispose()
        {
            GC.SuppressFinalize(this);
        }

        public bool MoveMany(long count)
        {
            _state = CursorState.Done;
            return false;
        }

        public bool MoveNext()
        {
            _state = CursorState.Done;
            return false;
        }

        public bool IsColumnActive(int col)
        {
            return _needCol(col);
        }

        /// <summary>
        /// The getter return the default value. A null getter usually fails the pipeline.
        /// </summary>
        public ValueGetter<TValue> GetGetter<TValue>(int col)
        {
            return (ref TValue value) =>
            {
                value = default(TValue);
            };
        }
    }
}
