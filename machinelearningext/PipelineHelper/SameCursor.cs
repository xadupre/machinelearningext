// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Combines an existing cursor with a schema not necessarily related.
    /// Used in <see cref="ScalerTransform"/>.
    /// </summary>
    public class SameCursor : IRowCursor
    {
        readonly IRowCursor _inputCursor;
        readonly ISchema _schema;

        public SameCursor(IRowCursor cursor, ISchema schema)
        {
            _schema = schema;
            _inputCursor = cursor;
        }

        public bool IsColumnActive(int col)
        {
            if (col < _inputCursor.Schema.ColumnCount)
                return _inputCursor.IsColumnActive(col);
            return false;
        }

        void IDisposable.Dispose()
        {
            _inputCursor.Dispose();
            GC.SuppressFinalize(this);
        }

        public ICursor GetRootCursor() { return this; }
        public ValueGetter<UInt128> GetIdGetter() { return _inputCursor.GetIdGetter(); }
        public CursorState State { get { return _inputCursor.State; } }
        public long Batch { get { return _inputCursor.Batch; } }
        public long Position { get { return _inputCursor.Position; } }
        public ISchema Schema { get { return _schema; } }
        public bool MoveMany(long count) { return _inputCursor.MoveMany(count); }
        public bool MoveNext() { return _inputCursor.MoveNext(); }
        public ValueGetter<TValue> GetGetter<TValue>(int col) { return _inputCursor.GetGetter<TValue>(col); }
    }
}
