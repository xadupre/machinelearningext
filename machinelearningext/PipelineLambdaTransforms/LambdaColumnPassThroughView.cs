﻿//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Scikit.ML.PipelineHelper;


namespace Scikit.ML.PipelineLambdaTransforms
{
    public static class LambdaColumnHelper
    {
        /// <summary>
        /// Modifies or add another column by applying a lambda column.
        /// </summary>
        /// <typeparam name="TSrc">input type</typeparam>
        /// <typeparam name="TDst">output type</typeparam>
        /// <param name="env">environment</param>
        /// <param name="name">name to register</param>
        /// <param name="input">input view</param>
        /// <param name="src">name of the input column</param>
        /// <param name="dst">name of the output column</param>
        /// <param name="typeSrc">input column type</param>
        /// <param name="typeDst">output column type</param>
        /// <param name="mapper">mapper to apply</param>
        /// <returns>IDataView</returns>
        public static IDataView Create<TSrc, TDst>(IHostEnvironment env, string name, IDataView input,
                        string src, string dst, ColumnType typeSrc, ColumnType typeDst,
                        ValueMapper<TSrc, TDst> mapper)
        {
            return new LambdaColumnPassThroughView<TSrc, TDst>(env, name, input, src, dst, typeSrc, typeDst, mapper);
        }
    }

    /// <summary>
    /// This implement a view which runs a lambda function.
    /// The view is more simple than <see cref="LambdaColumnTransform"/>
    /// and tends to make it as a pass-through transform.
    /// </summary>
    public class LambdaColumnPassThroughView<TSrc, TDst> : IDataView
    {
        readonly IDataView _source;
        readonly string _columnSrc;
        readonly string _columnDst;
        readonly ColumnType _typeSrc;
        readonly ColumnType _typeDst;
        readonly ValueMapper<TSrc, TDst> _mapper;
        readonly Schema _newSchema;
        readonly IHost _host;
        readonly int _srcIndex;

        public IDataView SourceTags { get { return _source; } }
        protected IHost Host { get { return _host; } }

        public LambdaColumnPassThroughView(IHostEnvironment env, string name, IDataView input,
                        string src, string dst, ColumnType typeSrc, ColumnType typeDst,
                        ValueMapper<TSrc, TDst> mapper)
        {
            _host = env.Register(name);
            _source = input;
            _mapper = mapper;
            _columnDst = dst;
            _columnSrc = src;
            _typeDst = typeDst;
            _typeSrc = typeSrc;
            _newSchema = Schema.Create(new ExtendedSchema(_source.Schema, new[] { dst }, new[] { typeDst }));
            _srcIndex = SchemaHelper.GetColumnIndex(_source.Schema, _columnSrc);
            _host.Except("Unable to find column '{0}' in input schema.", _columnSrc);
        }

        public bool CanShuffle
        {
            get { return _source.CanShuffle; }
        }

        public Schema Schema
        {
            get { return _newSchema; }
        }

        public long? GetRowCount()
        {
            return _source.GetRowCount();
        }

        /// <summary>
        /// When the last column is requested, we also need the column used to compute it.
        /// This function ensures that this column is requested when the last one is.
        /// </summary>
        bool PredicatePropagation(int col, Func<int, bool> predicate)
        {
            if (predicate(col))
                return true;
            if (col == _srcIndex)
                return predicate(_source.Schema.Count);
            return predicate(col);
        }

        public RowCursor GetRowCursor(Func<int, bool> predicate, Random rand = null)
        {
            if (predicate(_source.Schema.Count))
            {
                var cursor = _source.GetRowCursor(i => PredicatePropagation(i, predicate), rand);
                return new LambdaCursor(this, cursor);
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return new SameCursor(_source.GetRowCursor(predicate, rand), Schema);
        }

        public RowCursor[] GetRowCursorSet(Func<int, bool> predicate, int n, Random rand = null)
        {
            if (predicate(_source.Schema.Count))
            {
                var cursors = _source.GetRowCursorSet(i => PredicatePropagation(i, predicate), n, rand);
                return cursors.Select(c => new LambdaCursor(this, c)).ToArray();
            }
            else
                // The new column is not required. We do not need to compute it. But we need to keep the same schema.
                return _source.GetRowCursorSet(predicate, n, rand)
                              .Select(c => new SameCursor(c, Schema))
                              .ToArray();
        }

        public class LambdaCursor : RowCursor
        {
            readonly LambdaColumnPassThroughView<TSrc, TDst> _view;
            readonly RowCursor _inputCursor;

            public LambdaCursor(LambdaColumnPassThroughView<TSrc, TDst> view, RowCursor cursor)
            {
                _view = view;
                _inputCursor = cursor;
            }

            public override RowCursor GetRootCursor()
            {
                return this;
            }

            public override bool IsColumnActive(int col)
            {
                if (col < _inputCursor.Schema.Count)
                    return _inputCursor.IsColumnActive(col);
                return true;
            }

            public override ValueGetter<RowId> GetIdGetter()
            {
                var getId = _inputCursor.GetIdGetter();
                return (ref RowId pos) =>
                {
                    getId(ref pos);
                };
            }

            public override CursorState State { get { return _inputCursor.State; } }
            public override long Batch { get { return _inputCursor.Batch; } }
            public override long Position { get { return _inputCursor.Position; } }
            public override Schema Schema { get { return _view.Schema; } }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                    _inputCursor.Dispose();
                GC.SuppressFinalize(this);
            }

            public override bool MoveMany(long count)
            {
                return _inputCursor.MoveMany(count);
            }

            public override bool MoveNext()
            {
                return _inputCursor.MoveNext();
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                if (col < _view.SourceTags.Schema.Count)
                    return _inputCursor.GetGetter<TValue>(col);
                else if (col == _view.SourceTags.Schema.Count)
                    return GetLambdaGetter() as ValueGetter<TValue>;
                else
                    throw _view.Host.Except("Column index {0} does not exist.", col);
            }

            protected ValueGetter<TDst> GetLambdaGetter()
            {
                var getter = _inputCursor.GetGetter<TSrc>(_view._srcIndex);
                TSrc temp = default(TSrc);
                return (ref TDst dst) =>
                {
                    getter(ref temp);
                    _view._mapper(in temp, ref dst);
                };
            }
        }
    }
}
