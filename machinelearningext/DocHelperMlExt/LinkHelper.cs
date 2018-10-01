// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;


namespace Scikit.ML.DocHelperMlExt
{
    public static class LinkHelper
    {
        public static void _Immutable()
        {
            var res = new ImmutableArray<float>();
            if (res == null)
                throw new Exception("No immutable");
        }
    }
}
