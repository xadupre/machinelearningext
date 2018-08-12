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
    /// Implements Optics algorithms.
    /// </summary>
    public class Optics
    {
        #region Fields

        public const int NOISE = -1;
        private readonly IReadOnlyCollection<IPointIdFloat> points;
        internal readonly KdTree kdt;

        #endregion

        public Optics(List<IPointIdFloat> points, int? seed = null)
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
        /// <returns>A total ordering of the points according to their density distribution.</returns>
        public OpticsOrdering Ordering(float epsilon, int minPoints, bool shuffle = false,
            int? seed = null, Action<string> onShuffle = null, Action onPointProcessing = null)
        {
            onShuffle = onShuffle ?? (s => { });
            onPointProcessing = onPointProcessing ?? (() => { });

            if (epsilon <= 0)
                throw new ArgumentException(String.Format("Argument epsilon must be positive. Got {0}", epsilon));
            if (minPoints <= 0)
                throw new ArgumentException(String.Format("Argument minPoints must be positive. Got {0}", minPoints));

            var ordering = new List<IPointIdFloat>();
            var orderingMapping = new Dictionary<long, long>();
            var reachabilityDistances = new Dictionary<long, float?>();
            var coreDistancesCache = new Dictionary<long, float?>();

            HashSet<long> processed = new HashSet<long>();
            int currentIndex = 0;

            Action<IPointIdFloat> ProcessPoint = p => MarkProcessed(p, orderingMapping, ordering, processed, ref currentIndex);

            IList<IPointIdFloat> pts;

            if (shuffle)
            {
                onShuffle("Shuffle");

                var rnd = seed.HasValue ? new Random(seed.Value) : new Random();
                pts = PointIdFloat.RandomShuffle(points, rnd);
            }
            else
                pts = points.ToList();

            foreach (var p in points)
            {
                onPointProcessing();

                long pId = p.id;
                if (!processed.Contains(pId))
                {
                    ProcessPoint(p);

                    double? coreDistance = CoreDistance(kdt, coreDistancesCache, p, epsilon, minPoints);

                    if (coreDistance != null)
                    {
                        var seeds = new PriorityQueue<float, IPointIdFloat>();
                        Update(ref seeds, processed, reachabilityDistances, coreDistancesCache, p, epsilon, minPoints);

                        while (!seeds.IsEmpty)
                        {
                            KeyValuePair<float, IPointIdFloat> kvp;
                            seeds.TryDequeue(out kvp);
                            var q = kvp.Value;
                            ProcessPoint(q);
                            coreDistance = CoreDistance(kdt, coreDistancesCache, q, epsilon, minPoints);
                            if (coreDistance != null)
                                Update(ref seeds, processed, reachabilityDistances,
                                       coreDistancesCache, q, epsilon, minPoints);
                        }
                    }
                }
            }
            return new OpticsOrdering(kdt, ordering, orderingMapping,
                                      reachabilityDistances, coreDistancesCache, epsilon, minPoints);
        }

        #endregion

        #region Private

        internal static void MarkProcessed(IPointIdFloat p, Dictionary<long, long> orderingMapping,
                                           List<IPointIdFloat> ordering, HashSet<long> processed, ref int currentIndex)
        {
            processed.Add(p.id);
            orderingMapping.Add(p.id, currentIndex);
            ordering.Add(p);
            currentIndex += 1;
        }

        internal static IList<IPointIdFloat> EpsilonNeighbourhood(KdTree kdt, IPointIdFloat p, float epsilon)
        {
            return kdt.PointsWithinDistance(p, epsilon);
        }

        internal static IList<IPointIdFloat> NNearestNeighbours(KdTree kdt, IPointIdFloat p, int minPoints)
        {
            return kdt.NearestNNeighbors(p, minPoints);
        }


        internal float? CoreDistance(KdTree kdt, Dictionary<long, float?> coreDistances,
                                      IPointIdFloat p, float eps, int minPoints)
        {
            float? distance;
            long id = p.id;
            if (coreDistances.ContainsKey(id))
                coreDistances.TryGetValue(id, out distance);
            else
            {
                var nNeighbours = NNearestNeighbours(kdt, p, minPoints);
                //There will always be at least one element, i.e. p itself
                float nthDistance = nNeighbours.Last().DistanceTo(p);

                if (nNeighbours.Count < minPoints || nthDistance > eps)
                    distance = null;
                else
                    distance = nthDistance;
                coreDistances.Add(id, distance);
            }
            return distance;
        }
        internal float? CoreDistance(KdTree kdt, IPointIdFloat p, float eps, int minPoints)
        {
            float? distance;
            long id = p.id;
            var nNeighbours = NNearestNeighbours(kdt, p, minPoints);
            //There will always be at least one element, i.e. p itself
            var nthDistance = nNeighbours.Last().DistanceTo(p);

            if (nNeighbours.Count < minPoints || nthDistance > eps)
                distance = null;
            else
                distance = nthDistance;
            return distance;
        }

        internal void expandClusterOrder(IPointIdFloat p)
        {
            throw new NotImplementedException();
        }

        internal void Update(
            ref PriorityQueue<float, IPointIdFloat> seeds,
            HashSet<long> processed,
            Dictionary<long, float?> reachabilityDistances,
            Dictionary<long, float?> coreDistancesCache,
            IPointIdFloat p,
            float eps,
            int minPoints)
        {
            IList<IPointIdFloat> neighbours = EpsilonNeighbourhood(kdt, p, eps);
            foreach (var o in neighbours)
            {

                long oId = o.id;
                if (!processed.Contains(oId))
                {
                    float? coreDistance;
                    coreDistancesCache.TryGetValue(p.id, out coreDistance);

                    float? reachDist;
                    // INVARIANT: coreDistance != null
                    float newReachDist = Math.Max(coreDistance.Value, o.DistanceTo(p));

                    reachabilityDistances.TryGetValue(oId, out reachDist);
                    if (reachDist == null || newReachDist < reachDist)
                    {   // reachDist == null => o is not in Seeds
                        // newReachDist < reachDist => o in Seeds, check for improvement
                        reachabilityDistances[oId] = newReachDist;
                        if (reachDist == null)
                            seeds.Enqueue(newReachDist, o);
                    }
                }
            }
        }


        #endregion
    }
}
