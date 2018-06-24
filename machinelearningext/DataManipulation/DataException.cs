// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Raised when there is a type mismatch.
    /// </summary>
    public class DataTypeError : Exception
    {
        public DataTypeError(string msg) : base(msg)
        {
        }
    }
}
