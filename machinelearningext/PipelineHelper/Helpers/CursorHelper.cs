// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
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
                var getter = GetColumnGetter(cur, i, sch);
                if (getter == null)
                    throw Contracts.Except($"Unable to get getter for column {i} from schema\n{SchemaHelper.ToString(sch)}.");
                res.Add(getter);
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
                    case DataKind.BL: return (Delegate)cur.GetGetter<VBufferEqSort<bool>>(col) ?? cur.GetGetter<VBuffer<bool>>(col);
                    case DataKind.I4: return (Delegate)cur.GetGetter<VBufferEqSort<int>>(col) ?? cur.GetGetter<VBuffer<int>>(col);
                    case DataKind.U4: return (Delegate)cur.GetGetter<VBufferEqSort<uint>>(col) ?? cur.GetGetter<VBuffer<uint>>(col);
                    case DataKind.I8: return (Delegate)cur.GetGetter<VBufferEqSort<Int64>>(col) ?? cur.GetGetter<VBuffer<Int64>>(col);
                    case DataKind.R4: return (Delegate)cur.GetGetter<VBufferEqSort<float>>(col) ?? cur.GetGetter<VBuffer<float>>(col);
                    case DataKind.R8: return (Delegate)cur.GetGetter<VBufferEqSort<double>>(col) ?? cur.GetGetter<VBuffer<double>>(col);
                    case DataKind.TX:
                        {
                            var res4 = cur.GetGetter<VBufferEqSort<DvText>>(col);
                            if (res4 != null)
                                return res4;
                            var res2 = cur.GetGetter<VBuffer<DvText>>(col);
                            if (res2 != null)
                                return res2;
                            var res = cur.GetGetter<VBuffer<ReadOnlyMemory<char>>>(col);
                            if (res != null)
                                return res;
                            return null;
                        }
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
            else
            {
                switch (colType.RawKind)
                {
                    case DataKind.BL: return cur.GetGetter<bool>(col);
                    case DataKind.I4: return cur.GetGetter<int>(col);
                    case DataKind.U4: return cur.GetGetter<uint>(col);
                    case DataKind.I8: return cur.GetGetter<Int64>(col);
                    case DataKind.R4: return cur.GetGetter<float>(col);
                    case DataKind.R8: return cur.GetGetter<double>(col);
                    case DataKind.TX:
                        {
                            var res2 = cur.GetGetter<DvText>(col);
                            if (res2 != null)
                                return res2;
                            var res = cur.GetGetter<ReadOnlyMemory<char>>(col);
                            if (res != null)
                                return res;
                            return null;
                        }
                    default:
                        throw new NotImplementedException(string.Format("Not implemented for kind {0}", colType));
                }
            }
        }
    }
}
