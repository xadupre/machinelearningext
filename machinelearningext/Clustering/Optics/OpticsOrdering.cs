// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Scikit.ML.NearestNeighbors;


namespace Scikit.ML.Clustering
{
    public class OpticsOrdering
    {

        public IReadOnlyCollection<IPointIdFloat> ordering => _ordering.AsReadOnly();
        public IReadOnlyDictionary<long, long> orderingMapping => _orderingMapping;
        private List<IPointIdFloat> _ordering;
        private Dictionary<long, long> _orderingMapping;

        private KdTree kdt;
        internal IReadOnlyDictionary<long, float?> reachabilityDistances;
        internal IReadOnlyDictionary<long, float?> coreDistancesCache;
        private double epsilon;
        private int minPoints;

        public OpticsOrdering(
            KdTree kdt,
            List<IPointIdFloat> ordering,
            Dictionary<long, long> orderingMapping,
            Dictionary<long, float?> reachabilityDistances,
            Dictionary<long, float?> coreDistancesCache,
            float epsilon,
            int minPoints)
        {
            this.kdt = kdt;
            this._ordering = ordering;
            this._orderingMapping = orderingMapping;
            this.coreDistancesCache = coreDistancesCache;
            this.reachabilityDistances = reachabilityDistances;
            this.epsilon = epsilon;
            this.minPoints = minPoints;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="eps">The value of epsilon', i.e. the radius to use for clustering</param>
        /// <returns></returns>
        public Dictionary<long, int> Cluster(float eps)
        {
            if (eps <= 0 || eps > epsilon)
                throw new ArgumentException(String.Format("Argument eps ({0}) must be positive and no larger than the original epsilon used for ordering ({1})", eps, epsilon));
            Dictionary<long, int> clusters = new Dictionary<long, int>();

            // Precondition: ε ≤ generating dist ε' used for ordering

            int clusterId = Optics.NOISE;

            foreach (var p in ordering)
            {
                if (reachabilityDistances.ContainsKey(p.id) && reachabilityDistances[p.id] <= eps)
                {
                    // Object.reachability_distance ≤ ε’
                    clusters[p.id] = clusterId;
                }
                else
                {
                    double? cd = coreDistancesCache[p.id];
                    if (cd.HasValue && cd.Value <= eps)
                    {
                        clusterId += 1;
                        clusters[p.id] = clusterId;
                    }
                    else
                        clusters[p.id] = Optics.NOISE;
                }
            }
            return clusters;
        }
    }
}
