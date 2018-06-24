// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Ext.DataManipulation
{
    public static class DataFrameOperationHelper
    {
        public static NumericColumn Addition(NumericColumn c1, NumericColumn c2)
        {
            switch (c1.Kind)
            {
                case DataKind.I4:
                    switch (c2.Kind)
                    {
                        case DataKind.I4:
                            {
                                var c1o = c1.Column as DataColumn<DvInt4>;
                                var c2o = c2.Column as DataColumn<DvInt4>;
                                if (c1 == null || c2 == null)
                                    throw new DataTypeError(string.Format("Addition not implemented for {0}, {1}.", c1.Kind, c2.Kind));
                                var res = new DataColumn<DvInt4>(c1.Length);
                                var a = c1o.Data;
                                var b = c2o.Data;
                                for (int i = 0; i < res.Length; ++i)
                                    res.Set(i, a[i] + b[i]);
                                return new NumericColumn(res);
                            }
                        default:
                            throw new DataTypeError(string.Format("Addition not implemented for {0}, {1}.", c1.Kind, c2.Kind));
                    }
                default:
                    throw new DataTypeError(string.Format("Addition not implemented for {0} for left element.", c1.Kind));
            }
        }
    }
}
