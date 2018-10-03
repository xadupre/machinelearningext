// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Immutable;


namespace Scikit.ML.DocHelperMlExt
{
    public static class LinkHelper
    {
        public static void _Immutable()
        {
            var res = ImmutableArray.Create<float>(0f);
            if (res.Length == 0)
                throw new Exception("No immutable");
        }
    }
}
