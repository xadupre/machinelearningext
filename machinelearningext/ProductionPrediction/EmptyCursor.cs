// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.ProductionPrediction
{
    public class EmptyCursor : RowCursor
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

        public override CursorState State { get { return _state; } }
        public override RowCursor GetRootCursor() { return this; }
        public override long Batch { get { return 0; } }
        public override long Position { get { return 0; } }
        public override Schema Schema { get { return _view.Schema; } }
        public override ValueGetter<RowId> GetIdGetter() { return (ref RowId uid) => { uid = new RowId(0, 1); }; }

        protected override void Dispose(bool disposing)
        {
            GC.SuppressFinalize(this);
        }

        public override bool MoveMany(long count)
        {
            _state = CursorState.Done;
            return false;
        }

        public override bool MoveNext()
        {
            _state = CursorState.Done;
            return false;
        }

        public override bool IsColumnActive(int col)
        {
            return _needCol(col);
        }

        /// <summary>
        /// The getter return the default value. A null getter usually fails the pipeline.
        /// </summary>
        public override ValueGetter<TValue> GetGetter<TValue>(int col)
        {
            return (ref TValue value) =>
            {
                value = default(TValue);
            };
        }
    }
}
