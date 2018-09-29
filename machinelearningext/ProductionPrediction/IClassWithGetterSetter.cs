// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.ProductionPrediction
{
    public delegate void ValueGetterInstance<TRow, TValue>(ref TRow row, ref TValue value);

    public interface IClassWithGetter<TRow>
    {
        // Delegate is of type ValueGetterInstance<TRow, TValue>;
        Delegate GetGetter(int col);
    }

    public interface IClassWithSetter<TRow>
    {
        Delegate[] GetCursorGetter(IRowCursor cursor);
        void Set(Delegate[] delegates);
    }

    public static class GetterSetterHelper
    {
        public static Dictionary<string, Delegate> GetGetter<TRow>()
            where TRow : IClassWithGetter<TRow>, new()
        {
            var inst = new TRow();
            var schema = SchemaDefinition.Create(typeof(TRow), SchemaDefinition.Direction.Read);
            var res = new Dictionary<string, Delegate>();
            for (int i = 0; i < schema.Count; ++i)
            {
                var name = schema[i].ColumnName;
                res[name] = inst.GetGetter(i);
            }
            return res;
        }
    }
}
