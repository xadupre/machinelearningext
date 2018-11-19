//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.PipelineGraphTransforms
{
    /// <summary>
    /// Similar to OpaqueDataView implementation. It provides a barrier for data pipe optimizations.
    /// Used in cross validatation to generate the train/test pipelines for each fold.
    /// However, it gives access to previous tag. The class can overwrite a schema.
    /// </summary>
    public sealed class SemiOpaqueDataView : IDataView
    {
        private readonly IDataView _source;
        private readonly Schema _newSchema;

        public IDataView SourceTags { get { return _source; } }

        public SemiOpaqueDataView(IDataView source, Schema newSchema = null)
        {
            _source = source;
            _newSchema = newSchema;
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public Schema Schema
        {
            get { return _newSchema == null ? _source.Schema : _newSchema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
        {
            return _source.GetRowCursor(predicate, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            return _source.GetRowCursorSet(out consolidator, predicate, n, rand);
        }
    }
}
