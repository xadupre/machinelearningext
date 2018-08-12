// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Similar to OpaqueDataView implementation. It provides a view which changes the type of one column
    /// but does not change the cursor. We assume the cursor is able to return the same column in
    /// a different type (so GetGetter&lt;TYPE&gt;(col) returns two different getters on TYPE).
    /// </summary>
    public sealed class TypeReplacementDataView : IDataView
    {
        private readonly IDataView _source;
        private readonly ISchema _schema;

        public IDataView SourceTags { get { return _source; } }

        public TypeReplacementDataView(IDataView source, TypeReplacementSchema newSchema)
        {
            _source = source;
            _schema = newSchema;
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public ISchema Schema
        {
            get { return _schema; }
        }

        public long? GetRowCount(bool lazy = true)
        {
            return _source.GetRowCount(lazy);
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            var res = new TypeReplacementCursor(_source.GetRowCursor(predicate, rand), Schema);
#if(DEBUG)
            if (!SchemaHelper.CompareSchema(_schema, res.Schema))
                SchemaHelper.CompareSchema(_schema, res.Schema, true);
#endif
            return res;
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            return _source.GetRowCursorSet(out consolidator, predicate, n, rand).Select(c => new TypeReplacementCursor(c, Schema)).ToArray();
        }

        class TypeReplacementCursor : IRowCursor
        {
            IRowCursor _cursor;
            ISchema _schema;

            public TypeReplacementCursor(IRowCursor cursor, ISchema newSchema)
            {
                _cursor = cursor;
                _schema = newSchema;
            }

            public ISchema Schema { get { return _schema; } }
            public ICursor GetRootCursor() { return this; }
            public bool IsColumnActive(int col) { return _cursor.IsColumnActive(col); }
            public ValueGetter<UInt128> GetIdGetter() { return _cursor.GetIdGetter(); }
            public CursorState State { get { return _cursor.State; } }
            public long Batch { get { return _cursor.Batch; } }
            public long Position { get { return _cursor.Position; } }
            public bool MoveMany(long count) { return _cursor.MoveMany(count); }
            public bool MoveNext() { return _cursor.MoveNext(); }
            public ValueGetter<TValue> GetGetter<TValue>(int col) { return _cursor.GetGetter<TValue>(col); }

            void IDisposable.Dispose()
            {
                _cursor.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
