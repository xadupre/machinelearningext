// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Data;
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
        private readonly Schema _schema;

        public IDataView SourceTags { get { return _source; } }

        public TypeReplacementDataView(IDataView source, TypeReplacementSchema newSchema)
        {
            _source = source;
            _schema = Schema.Create(newSchema);
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public Schema Schema
        {
            get { return _schema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        public RowCursor GetRowCursor(Func<int, bool> predicate, Random rand = null)
        {
            var res = new TypeReplacementCursor(_source.GetRowCursor(predicate, rand), Schema);
#if(DEBUG)
            if (!SchemaHelper.CompareSchema(_schema, res.Schema))
                SchemaHelper.CompareSchema(_schema, res.Schema, true);
#endif
            return res;
        }

        public RowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, Random rand = null)
        {
            return _source.GetRowCursorSet(out consolidator, predicate, n, rand)
                          .Select(c => new TypeReplacementCursor(c, Schema)).ToArray();
        }

        class TypeReplacementCursor : RowCursor
        {
            RowCursor _cursor;
            Schema _schema;

            public TypeReplacementCursor(RowCursor cursor, ISchema newSchema)
            {
                _cursor = cursor;
                _schema = Schema.Create(newSchema);
            }

            public override Schema Schema { get { return _schema; } }
            public override RowCursor GetRootCursor() { return this; }
            public override bool IsColumnActive(int col) { return _cursor.IsColumnActive(col); }
            public override ValueGetter<RowId> GetIdGetter() { return _cursor.GetIdGetter(); }
            public override CursorState State { get { return _cursor.State; } }
            public override long Batch { get { return _cursor.Batch; } }
            public override long Position { get { return _cursor.Position; } }
            public override bool MoveMany(long count) { return _cursor.MoveMany(count); }
            public override bool MoveNext() { return _cursor.MoveNext(); }
            public override ValueGetter<TValue> GetGetter<TValue>(int col) { return _cursor.GetGetter<TValue>(col); }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _cursor.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
