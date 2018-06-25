// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Data;
//using Microsoft.ML.Runtime.Data.Conversion;


namespace Microsoft.ML.Ext.DataManipulation
{
    public static class ObjectConversion
    {
        public static void Convert<T>(object src, out T value)
        {
            if (src is string)
            {
                var dv = new DvText((string)src);
                value = (T)(object)dv;
            }
            else
                value = (T)src;
        }
    }
}
