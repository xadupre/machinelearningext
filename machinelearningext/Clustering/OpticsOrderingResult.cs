// See the LICENSE file in the project root for more information.

namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Ordering of a row after OPTICS.
    /// </summary>
    public class OpticsOrderingResult
    {
        /// <summary>
        /// index of the row in OPTICS ordering
        /// </summary>
        public long id;

        /// <summary>
        /// reachability distance
        /// </summary>
        public float reachability;

        /// <summary>
        /// core distance
        /// </summary>
        public float core;
    }
}
