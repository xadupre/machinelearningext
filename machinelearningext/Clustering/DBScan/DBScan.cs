// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Scikit.ML.NearestNeighbors;

[assembly: InternalsVisibleTo("TestMachineLearningExt")]
namespace Scikit.ML.Clustering
{
    /// <summary>
    /// Implements DBScan algorithm.
    /// </summary>
    public class DBScan
    {
        #region Fields

        public const int NOISE = -1;
        private readonly IReadOnlyCollection<IPointIdFloat> points;
        private KdTree kdt;

        #endregion

        public DBScan(List<IPointIdFloat> points, int? seed = null)
        {
            this.points = points.AsReadOnly();
            this.kdt = new KdTree(this.points, seed: seed);
        }

        #region API

        /// <summary>
        /// 
        /// </summary>
        /// <param name="epsilon"></param>
        /// <param name="minPoints"></param>
        /// <param name="shuffle"></param>
        /// <param name="seed"></param>
        /// <param name="onShuffle"></param>
        /// <param name="onPointProcessing"></param>
        /// <returns>A dictionary storing, for each point's id, the cluster it is assigned to.</returns>
        public Dictionary<long, int> Cluster(float epsilon, int minPoints, bool shuffle = false,
                                             int? seed = null, Action<string> onShuffle = null,
                                             Action<int> onPointProcessing = null)
        {
            onShuffle = onShuffle ?? (s => { });
            onPointProcessing = onPointProcessing ?? (c => { });

            if (epsilon <= 0)
                throw new ArgumentException(String.Format("Argument epsilon must be positive. Got {0}", epsilon));

            if (minPoints <= 0)
                throw new ArgumentException(String.Format("Argument minPoints must be positive. Got {0}", minPoints));

            Dictionary<long, int> clusters = new Dictionary<long, int>();
            HashSet<long> processed = new HashSet<long>();
            int C = 0;

            IList<IPointIdFloat> pts;

            if (shuffle)
            {
                onShuffle("Shuffle");
                var rnd = seed.HasValue ? new Random(seed.Value) : new Random();
                pts = PointIdFloat.RandomShuffle(points, rnd);
            }
            else
                pts = points.ToList();

            int nbtotal = points.Count;

            foreach (var p in points)
            {
                onPointProcessing(C);

                if (!processed.Contains(p.id))
                {
                    processed.Add(p.id);
                    var neighbours = kdt.PointsWithinDistance(p, epsilon);
                    if (neighbours.Count() < minPoints)
                        clusters.Add(p.id, NOISE);
                    else
                    {
                        C += 1;
                        ExpandCluster(clusters, processed, pts, p, neighbours.ToList(), C, epsilon, minPoints);
                    }
                }
            }
            return clusters;
        }

        public IList<IPointIdFloat> RegionQuery(IPointIdFloat p, float epsilon)
        {
            return RegionQuery(kdt, p, epsilon);
        }

        public float Score(IPointIdFloat p, float epsilon, Dictionary<long, int> mapClusters)
        {
            var res = RegionQuery(kdt, p, epsilon);
            if (res.Count() <= 1)
                return 1f;
            else
            {
                var sorted = res.Select(pe => new Tuple<float, int>((float)(1 / (epsilon + p.DistanceTo(pe))), mapClusters[pe.id]))
                                .OrderBy(c => c);
                float score = 0f;
                int last = mapClusters[p.id];
                float lastd = 0f;
                foreach (var el in sorted)
                {
                    if (el.Item2 != last)
                    {
                        score += el.Item1 - lastd;
                        lastd = el.Item1;
                    }
                }
                return score;
            }
        }

        #endregion

        #region Private

        /**
         * 
         */
        internal static IList<IPointIdFloat> RegionQuery(IList<IPointIdFloat> points, IPointIdFloat p, float epsilon)
        {
            return points.Where(x => p.DistanceTo(x) <= epsilon).ToList();
        }


        /**
         * 
         */
        internal static IList<IPointIdFloat> RegionQuery(KdTree points, IPointIdFloat p, float epsilon)
        {
            return points.PointsWithinDistance(p, epsilon);
        }


        /**
         * 
         */
        internal static void ExpandCluster(Dictionary<long, int> clusters, HashSet<long> processed,
            IList<IPointIdFloat> points, IPointIdFloat pAdd, List<IPointIdFloat> pNeighbours,
            int clusterId, float epsilon, int minPoints)
        {
            if (!clusters.ContainsKey(pAdd.id))
            {
                // Some points might have been added as neighbours even though they were not visited.
                // Line A below.
                clusters.Add(pAdd.id, clusterId);
            }

            for (int i = 0; i < pNeighbours.Count; i++)
            {
                IPointIdFloat q = pNeighbours[i];
                if (!processed.Contains(q.id))
                {
                    processed.Add(q.id);
                    var qNeighbours = RegionQuery(points, q, epsilon);
                    if (qNeighbours.Count() > minPoints)
                    {
                        pNeighbours.AddRange(qNeighbours);
                    }
                }
                if (!clusters.ContainsKey(q.id))
                {
                    clusters.Add(q.id, clusterId);
                }
            }
        }
        #endregion
    }
}
