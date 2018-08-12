// See the LICENSE file in the project root for more information.

namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Cluster in which a row belongs to after a clustering algorithm.
    /// </summary>
    public class ClusteringResult
    {
        /// <summary>
        /// cluster of the row
        /// </summary>
        public int cl;

        /// <summary>
        /// score or probability
        /// </summary>
        public float score;
    }
}
