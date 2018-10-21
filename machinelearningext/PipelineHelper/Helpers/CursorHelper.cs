// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Helpers about DataView.
    /// </summary>
    public static class CursorHelper
    {
        /// <summary>
        /// Returns the getters for all columns.
        /// </summary>
        public static Delegate[] GetAllGetters(IRowCursor cur)
        {
            var sch = cur.Schema;
            var res = new List<Delegate>();
            for (int i = 0; i < sch.ColumnCount; ++i)
            {
                if (sch.IsHidden(i))
                    continue;
                res.Add(GetColumnGetter(cur, i, sch));
            }
            return res.ToArray();
        }

        public static Delegate GetColumnGetter(IRowCursor cur, int col, Schema sch = null)
        {
            if (sch == null)
                sch = cur.Schema;
            var colType = sch.GetColumnType(col);
            if (colType.IsVector)
            {
                switch (colType.ItemType.RawKind)
                {
                    case DataKind.BL: return cur.GetGetter<bool>(col);
                    case DataKind.I4: return cur.GetGetter<int>(col);
                    case DataKind.U4: return cur.GetGetter<uint>(col);
                    case DataKind.I8: return cur.GetGetter<Int64>(col);
                    case DataKind.R4: return cur.GetGetter<float>(col);
                    case DataKind.R8: return cur.GetGetter<double>(col);
                    case DataKind.TX: return cur.GetGetter<ReadOnlyMemory<char>>(col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
            else
            {
                switch (colType.RawKind)
                {
                    case DataKind.BL: return cur.GetGetter<VBuffer<bool>>(col);
                    case DataKind.I4: return cur.GetGetter<VBuffer<int>>(col);
                    case DataKind.U4: return cur.GetGetter<VBuffer<uint>>(col);
                    case DataKind.I8: return cur.GetGetter<VBuffer<Int64>>(col);
                    case DataKind.R4: return cur.GetGetter<VBuffer<float>>(col);
                    case DataKind.R8: return cur.GetGetter<VBuffer<double>>(col);
                    case DataKind.TX: return cur.GetGetter<VBuffer<ReadOnlyMemory<char>>>(col);
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
        }
    }
}
