// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Ext.NearestNeighbours;


namespace TestMachineLearningExt
{
    [TestClass()]
    public class TestNearestNeighboursKdTree
    {
        private static Random rnd = new Random();
        private IPointIdFloat point1D = new PointIdFloat(0, new List<float>() { 1.0f });
        private IPointIdFloat point2D = new PointIdFloat(1, new List<float>() { -0.5f, 1.0f });
        private static readonly object syncLock = new object();

        #region common

        [TestMethod()]
        public void ClasRandomShuffleTest()
        {
            IList<IPointIdFloat> points = new List<IPointIdFloat>()
            {
                new PointIdFloat(0, new List<float>() { 1, 1, 1 }),
                new PointIdFloat(1, new List<float>() { 2, 1, 1 }),
                new PointIdFloat(2, new List<float>() { 1, 2, 1 }),
                new PointIdFloat(3, new List<float>() { 1, 1, 2 }),
                new PointIdFloat(4, new List<float>() { 3, 3, 3 }),
                new PointIdFloat(5, new List<float>() { 2, 2, 2 })
            };
            IList<IPointIdFloat> pointsCopy = new List<IPointIdFloat>(points);

            IList<IPointIdFloat> shuffledPoints = PointIdFloat.RandomShuffle(points.ToList().AsReadOnly(), new Random());
            Assert.IsTrue(SequenceEquivalent(shuffledPoints.ToList(), points.ToList(), PointIdFloat.PointsComparison), "shuffled list should have the same elements and multiplicity");
            Assert.IsFalse(points.SequenceEqual(shuffledPoints), "shuffled list should have a different order");
            Assert.IsTrue(points.SequenceEqual(pointsCopy), "Original list must not be changed");
        }

        public static bool SequenceEquivalent<T>(List<T> list1, List<T> list2, Comparison<T> comparer)
        {
            list1.Sort(comparer);
            list2.Sort(comparer);
            return list1.SequenceEqual(list2);
        }

        public static IPointIdFloat RandomPoint(Random rnd, int dimension = 2)
        {
            List<float> coordinates = new List<float>();
            for (int i = 0; i < dimension; i++)
                coordinates.Add(Int32.MaxValue * (float)rnd.NextDouble());
            return new PointIdFloat(0, coordinates);
        }

        #endregion

        #region points

        [TestMethod()]
        public void ClasPointContructorEnumerableTest()
        {
            IPointIdFloat p = new PointIdFloat(0, new List<float>() { 1.0f });
            Assert.AreEqual(1, p.dimension);
            Assert.IsTrue(p.coordinates.Values.SequenceEqual(new List<float>() { 1.0f }));

            p = new PointIdFloat(0, new List<float>() { 1.0f, -0.000031f, (float)Math.PI, (float)-Math.E });
            Assert.AreEqual(4, p.dimension);
            Assert.IsTrue(p.coordinates.Values.SequenceEqual(new List<float>() { 1.0f, -0.000031f, (float)Math.PI, (float)-Math.E }));
        }

        [TestMethod()]
        public void ClasPointContructorVarArgsTest()
        {
            PointIdFloat p = new PointIdFloat(0, 1.0f);
            Assert.AreEqual(1, p.dimension);
            Assert.IsTrue(p.coordinates.Values.SequenceEqual(new List<float>() { 1.0f }));

            p = new PointIdFloat(0, 1.0f, -0.000031f, (float)Math.PI, (float)-Math.E);
            Assert.AreEqual(4, p.dimension);
            Assert.IsTrue(p.coordinates.Values.SequenceEqual(new List<float>() { 1.0f, -0.000031f, (float)Math.PI, (float)-Math.E }));
            PointIdFloat p2 = new PointIdFloat(0, new List<float>() { 1.0f, -0.000031f, (float)Math.PI, (float)-Math.E });
            Assert.AreEqual(p, p2);
        }

        [TestMethod()]
        public void ClasdistanceToTest()
        {
            PointIdFloat x = new PointIdFloat(0, new List<float>() { 1.0f });
            PointIdFloat y = new PointIdFloat(1, new List<float>() { 2.0f });
            Assert.AreEqual(0.0f, VectorDistanceHelper.L2Norm(x.coordinates, x.coordinates));
            Assert.AreEqual(1.0f, VectorDistanceHelper.L2Norm(x.coordinates, y.coordinates));

            x = new PointIdFloat(2, new List<float>() { 1.0f, 2.2f });
            y = new PointIdFloat(3, new List<float>() { 4.0f, 6.2f });
            Assert.AreEqual(0.0f, VectorDistanceHelper.L2Norm(y.coordinates, y.coordinates));
            Assert.IsTrue(Math.Abs(3.535534f - VectorDistanceHelper.L2Norm(x.coordinates, y.coordinates)) < 1e-5);

            x = new PointIdFloat(5, new List<float>() { 1.0f, 2.5f, 3.0f });
            y = new PointIdFloat(6, new List<float>() { 2.0f, 3.0f, 3.0f });
            var d1 = 0.6454972f;
            var d2 = VectorDistanceHelper.L2Norm(x.coordinates, y.coordinates);
            Assert.IsTrue(Math.Abs(d1 - d2) < 1e-5);
        }

        [TestMethod()]
        public void ClasdistanceIsSimmetricToTest()
        {
            var x = new PointIdFloat(0, new List<float>() { 1.0f });
            var y = new PointIdFloat(1, new List<float>() { 2.0f });
            Assert.AreEqual(VectorDistanceHelper.L2Norm(x.coordinates, y.coordinates),
                            VectorDistanceHelper.L2Norm(y.coordinates, x.coordinates));
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Incompatible point dimensions: expected 1, got 2")]
        public void ClasdistanceToThrowTest()
        {
            var p1 = new PointIdFloat(0, new List<float>() { 1.0f });
            var p2 = new PointIdFloat(1, new List<float>() { 2.0f, 3.0f });
            VectorDistanceHelper.L2Norm(p1.coordinates, p2.coordinates);
        }

        [TestMethod()]
        public void ClasEqualsTest()
        {
            var p1 = new PointIdFloat(0, new List<float>() { 1.0f });
            var p2 = new PointIdFloat(1, new List<float>() { 2.0f, 3.0f });
            var p3 = new PointIdFloat(2, new List<float>() { 2.0f, 3.0f, 4.0f });
            var p4 = new PointIdFloat(3, new List<float>() { 2.0f, 3.0f, 4.0f });
            var p5 = new PointIdFloat(4, new List<float>() { 2.0f, 3.0f, 5.0f });
            var p6 = new PointIdFloat(5, new List<float>() { 2.0f, 3.00001f, 5.00001f });

            Assert.IsTrue(p1.Equals(p1));
            Assert.IsTrue(p2.Equals(p2));
            Assert.IsTrue(p3.Equals(p3));
            Assert.IsTrue(p4.Equals(p4));
            Assert.IsTrue(p5.Equals(p5));

            Assert.IsFalse(p1.Equals(p2));
            Assert.IsTrue(p3.Equals(p4));
            Assert.IsTrue(p4.Equals(p3));
            Assert.IsFalse(p4.Equals(p5));
            Assert.IsFalse(p6.Equals(p5));
        }

        [TestMethod()]
        public void ClasGetHashCodeTest()
        {
            var p1 = new PointIdFloat(0, new List<float>() { 1.0f });
            var p2 = new PointIdFloat(1, new List<float>() { 2.0f, 3.0f });
            var p3 = new PointIdFloat(2, new List<float>() { 2.0f, 3.0f, 4.0f });
            var p4 = new PointIdFloat(3, new List<float>() { 2.0f, 3.0f, 4.0f });
            var p5 = new PointIdFloat(4, new List<float>() { 2.0f, 3.0f, 5.0f });
            var p6 = new PointIdFloat(5, new List<float>() { 2.0f, 3.00001f, 5.00001f });
            var l1 = (new List<float>() { 1.0f, 2.0f }).GetHashCode();
            var l2 = (new List<float>() { 1.0f, 2.0f }).GetHashCode();

            Assert.IsFalse(p1.GetHashCode() == p2.GetHashCode());
            Assert.IsTrue(p3.GetHashCode() == p4.GetHashCode());
            Assert.IsFalse(p4.GetHashCode() == p5.GetHashCode());
            Assert.IsFalse(p5.GetHashCode() == p6.GetHashCode());
        }

        [TestMethod()]
        public void ClasToStringTest()
        {
            var p = new PointIdFloat(0, new List<float>() { 1.0f });
            var s1 = p.ToString();
            Assert.AreEqual("PointIdFloat1D(1.000)", s1);

            p = new PointIdFloat(1, new List<float>() { -1.0f, 0.55f });
            s1 = p.ToString();
            Assert.AreEqual("PointIdFloat2D(-1.000,0.550)", s1);
        }

        #endregion

        #region Constructor

        [TestMethod()]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClasKdTreeConstructorNullPointsListTest()
        {
            List<IPointIdFloat> x = null;
            new KdTree(x);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClasKdTreeConstructorNullPointsTest()
        {
            new KdTree(null, -1);
        }

        [TestMethod()]
        [ExpectedException(typeof(NullReferenceException))]
        public void ClasKdTreeConstructorNullPointTest()
        {
            new KdTree(null, point1D, null);
        }

        [TestMethod()]
        public void ClasKdTreeConstructorEmptyPointsListTest()
        {
            new KdTree(new List<PointIdFloat>());
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Points array must be non-empty and all points need to have the same dimension 1")]
        public void ClasKdTreeConstructorInconsistentPointsListTest()
        {
            new KdTree(new List<IPointIdFloat>() { point1D, point2D });
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Points array must be non-empty and all points need to have the same dimension 2")]
        public void ClasKdTreeConstructorInconsistentDimensionTest()
        {
            if (rnd.NextDouble() < 0.5)
                new KdTree(new List<IPointIdFloat>() { point1D }, 2);
            else
                new KdTree(new List<IPointIdFloat>() { point2D }, 1);
        }

        [TestMethod()]
        public void ClasKdTreeConstructorTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { point1D }, 1);
            Assert.AreEqual(1, kdt.dimension);

            kdt = new KdTree(new List<IPointIdFloat>() { point2D }, 2);
            Assert.AreEqual(2, kdt.dimension);

            kdt = new KdTree(null, point2D);
            Assert.AreEqual(2, kdt.dimension);
        }

        #endregion

        #region Count

        [TestMethod()]
        public void ClasCountTest()
        {
            IList<IPointIdFloat> points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                new PointIdFloat(1, 0f, 1f), new PointIdFloat(2, 3f, 3f),
                new PointIdFloat(3, 1.5f, 2f), new PointIdFloat(4, 5f, -1f) };
            KdTree kdt = new KdTree(points);
            Assert.AreEqual(2, kdt.dimension);
            Assert.AreEqual(points.Count(), kdt.Count());

            points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f, 0f),
                        new PointIdFloat(1, 0f, 1f, -1f), new PointIdFloat(2, 3f, 3f, -2f),
                        new PointIdFloat(3, 1.5f, 2f, -3f), new PointIdFloat(4, 5f, -1f, -4f) };
            kdt = new KdTree(points);
            Assert.AreEqual(3, kdt.dimension);
            Assert.AreEqual(points.Count(), kdt.Count());
        }

        #endregion

        #region Contains

        [TestMethod()]
        public void ClasContainsTest()
        {
            var points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                        new PointIdFloat(1, 0f, 1f), new PointIdFloat(2, 3f, 3f),
                        new PointIdFloat(3, 1.5f, 2f), new PointIdFloat(4, 5f, -1f) };
            KdTree kdt = new KdTree(points);
            Assert.AreEqual(2, kdt.dimension);
            Assert.IsTrue(points.All(p => kdt.Contains(p)));
            Assert.IsFalse(kdt.Contains(point2D));

            points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f, 0f), new PointIdFloat(1, 0f, 1f, -1f),
                new PointIdFloat(2, 3f, 3f, -2f), new PointIdFloat(3, 1.5f, 2f, -3f), new PointIdFloat(4, 5f, -1f, -4f) };

            kdt = new KdTree(points);
            Assert.AreEqual(3, kdt.dimension);
            Assert.IsTrue(points.All(p => kdt.Contains(p)));
            Assert.IsFalse(kdt.Contains(point2D));
            Assert.IsFalse(kdt.Contains(new PointIdFloat(1, 2, 3)));
        }

        #endregion

        #region Add

        [TestMethod()]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClasAddNullPointTest()
        {
            IList<IPointIdFloat> points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                new PointIdFloat(1, 0f, 1f), new PointIdFloat(2, 3f, 3f), new PointIdFloat(3, 1.5f, 2f),
                new PointIdFloat(4, 5f, -1f) };
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(5, 1f, 2f) });
            kdt.Add(null);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Wrong Point dimension: expected 2, got 3")]
        public void ClasAddWrongDimensionPointTest()
        {
            IList<IPointIdFloat> points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                new PointIdFloat(1, 0f, 1f), new PointIdFloat(2, 3f, 3f),
                new PointIdFloat(4, 1.5f, 2f), new PointIdFloat(3, 5f, -1f) };
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f) });
            kdt.Add(new PointIdFloat(1, 1f, 2f, 3f));
        }

        [TestMethod()]
        public void ClasAddTest()
        {
            IPointIdFloat p1 = new PointIdFloat(0, 1f, 2f);
            IPointIdFloat p2 = new PointIdFloat(1, -1f, 2f);
            IPointIdFloat p3 = new PointIdFloat(2, 1f, -2f);
            IPointIdFloat p1B = new PointIdFloat(3, 1f, 2f);
            IPointIdFloat p2B = new PointIdFloat(4, -1f, 2f);
            IPointIdFloat p2C = new PointIdFloat(5, -1f, 2f);
            IPointIdFloat p3B = new PointIdFloat(6, 1f, -2f);
            IPointIdFloat p4 = new PointIdFloat(7, 0f, 0f);

            KdTree kdt = new KdTree(new List<IPointIdFloat>() { p1 });
            Assert.AreEqual(1, kdt.Count());

            kdt.Add(p2);
            Assert.AreEqual(2, kdt.Count());
            Assert.IsTrue(kdt.Contains(p1));
            Assert.IsTrue(kdt.Contains(p2));

            kdt.Add(p3);
            Assert.AreEqual(3, kdt.Count());
            Assert.IsTrue(kdt.Contains(p1));
            Assert.IsTrue(kdt.Contains(p2));
            Assert.IsTrue(kdt.Contains(p3));

            kdt.Add(p1B);
            Assert.AreEqual(4, kdt.Count());
            Assert.IsTrue(kdt.Contains(p1));
            Assert.IsTrue(kdt.Contains(p1B));
            Assert.IsTrue(kdt.Contains(p2));
            Assert.IsTrue(kdt.Contains(p3));

            kdt.Add(p2B);
            kdt.Add(p3B);
            kdt.Add(p2C);
            Assert.AreEqual(7, kdt.Count());
            Assert.IsTrue(kdt.Contains(p1));
            Assert.IsTrue(kdt.Contains(p1B));
            Assert.IsTrue(kdt.Contains(p2));
            Assert.IsTrue(kdt.Contains(p2B));
            Assert.IsTrue(kdt.Contains(p2C));
            Assert.IsTrue(kdt.Contains(p3));
            Assert.IsTrue(kdt.Contains(p3B));
        }

        #endregion

        #region PointsWithinDistance

        [TestMethod()]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClasPointsWithinDistanceNullPointTest()
        {
            IList<IPointIdFloat> points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                        new PointIdFloat(1, 0f, 1f), new PointIdFloat(2, 3f, 3f),
                        new PointIdFloat(3, 1.5f, 2f), new PointIdFloat(4, 5f, -1f) };
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(6, 1f, 2f) });
            kdt.PointsWithinDistance(null, 1);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Wrong Point dimension: expected 2, got 3")]
        public void ClasPointsWithinDistanceWrongDimensionPointTest()
        {
            IList<IPointIdFloat> points = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                        new PointIdFloat(1, 0f, 1f), new PointIdFloat(2, 3f, 3f),
                        new PointIdFloat(3, 1.5f, 2f), new PointIdFloat(4, 5f, -1f) };
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(6, 1f, 2f) });
            kdt.PointsWithinDistance(new PointIdFloat(7, 1f, 2f, 3f), 1);
        }

        [TestMethod()]
        public void ClasPointsWithinDistanceTest()
        {
            IPointIdFloat p1 = new PointIdFloat(0, 1f, 2f);
            IPointIdFloat p2 = new PointIdFloat(1, 2f, 1f);
            IPointIdFloat p3 = new PointIdFloat(2, 200f, -100f);
            IPointIdFloat p4 = new PointIdFloat(3, 150f, -50f);

            List<IPointIdFloat> points2D = new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f),
                            new PointIdFloat(4, 0f, 1f), new PointIdFloat(3, 3f, 3f),
                            new PointIdFloat(5, 1.5f, 2f), new PointIdFloat(6, 5f, -1f) };
            List<IPointIdFloat> points3D = new List<IPointIdFloat>() { new PointIdFloat(0, 0f, 0f, 0f),
                    new PointIdFloat(1, -1f, 2f, 3.5f), new PointIdFloat(3, 0.55f, 30f, -10f),
                    new PointIdFloat(2, 1f, 1.2f, 1.5f), new PointIdFloat(4, 4.25f, 0.37f, 0f) };

            Func<IPointIdFloat, float> keySelector = t => KdTree.KdTreeNode.KeyByDepth(t, 0);

            //should return an empty Set if all the points are not within distance
            var kdt = new KdTree(null, p1);

            Func<IPointIdFloat, float, List<IPointIdFloat>, bool> AssertResult = (target, maxDistance, expectedResults) =>
            {

                var results = kdt.PointsWithinDistance(target, maxDistance);
                if (!SequenceEquivalent(expectedResults, results, keySelector))
                {
                    var distance = results.Select(c => VectorDistanceHelper.L2(c.coordinates, target.coordinates)).ToArray();
                    results = kdt.PointsWithinDistance(target, maxDistance);
                    Assert.IsTrue(SequenceEquivalent(expectedResults, results, keySelector));
                }
                results.ToList().ForEach(p =>
                {
                    //Console.WriteLine(string.Format("{0}, {1}", p.ToString(), 
                    //                    VectorDistanceHelper.L2(p.coordinates, target.coordinates)));
                    Assert.IsTrue(VectorDistanceHelper.L2(p.coordinates, target.coordinates) <= maxDistance);
                });
                return true;
            };

            var result = kdt.PointsWithinDistance(p3, 100);
            Assert.AreEqual(0, result.Count());

            kdt = new KdTree(null, p1, p2);
            result = kdt.PointsWithinDistance(p3, 100);
            Assert.AreEqual(0, result.Count());

            kdt = new KdTree(null, p1);
            AssertResult(p2, 10, new List<IPointIdFloat>() { p1 });

            AssertResult(p1, 1, new List<IPointIdFloat>() { p1 });

            //should include the points itself, if in the tree
            kdt = new KdTree(null, p1, p2);

            AssertResult(p2, 2, new List<IPointIdFloat>() { p1, p2 });

            AssertResult(p2, 0, new List<IPointIdFloat>() { p2 });

            IPointIdFloat p3D = points3D.First();
            kdt = new KdTree(null, p3D);

            AssertResult(p3D, 2, new List<IPointIdFloat>() { p3D });

            //should return all the points within distance
            kdt = new KdTree(points3D);

            AssertResult(new PointIdFloat(0, 0.5f, 0.6f, 0.7f), 2f, new List<IPointIdFloat>() { points3D[0], points3D[3] });
            AssertResult(new PointIdFloat(1, -5f, -5f, -5f), 1f, new List<IPointIdFloat>() { });
            AssertResult(new PointIdFloat(2, -5f, -5.5f, -6f), 12.5f, new List<IPointIdFloat>() { points3D[0], points3D[3], points3D[4] });
            AssertResult(new PointIdFloat(3, -0.95f, 2.05f, 3.45f), 0.1f, new List<IPointIdFloat>() { points3D[1] });
            AssertResult(new PointIdFloat(4, -0.9f, 2.05f, 3.45f), 4f, new List<IPointIdFloat>() { points3D[1], points3D[3] });
            kdt = new KdTree(points2D);
            AssertResult(new PointIdFloat(5, 1.1f, 1.9f), 1, new List<IPointIdFloat>() { points2D[0], points2D[3] });
            AssertResult(new PointIdFloat(6, 1.1f, 1.9f), 10, points2D);
            AssertResult(new PointIdFloat(7, -5f, 5f), 1, new List<IPointIdFloat>() { });
            AssertResult(new PointIdFloat(8, -5f, 5f), 8, new List<IPointIdFloat>() { points2D[0], points2D[1], points2D[3] });
            AssertResult(new PointIdFloat(9, -5f, 5f), 10, new List<IPointIdFloat>() { points2D[0], points2D[1], points2D[2], points2D[3] });
            AssertResult(new PointIdFloat(10, 3.01f, 2.9999f), 0.1f, new List<IPointIdFloat>() { points2D[2] });
            AssertResult(new PointIdFloat(11, 3.01f, 2.9999f), 2, new List<IPointIdFloat>() { points2D[2], points2D[3] });
            AssertResult(new PointIdFloat(12, 1.6f, 2f), 10, points2D);
            AssertResult(new PointIdFloat(13, 160f, 2f), 160, new List<IPointIdFloat>() { points2D[0], points2D[2], points2D[3], points2D[4] });
        }

        #endregion

        #region NearestNeighbour

        [TestMethod()]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClasNearestNeighbourNullPointTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f) });
            kdt.NearestNeighbour(null);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Wrong Point dimension: expected 2, got 3")]
        public void ClasNearestNeighbourWrongDimensionPointTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f) });
            kdt.NearestNeighbour(new PointIdFloat(0, 1f, 2f, 3f));
        }

        [TestMethod()]
        public void ClasNearestNeighbourTest()
        {
            var points2D = new List<IPointIdFloat>() {
                new PointIdFloat(0, 1f, 2f),
                new PointIdFloat(1, 0f, 1f),
                new PointIdFloat(2, 3f, 3f),
                new PointIdFloat(3, 1.5f, 2f),
                new PointIdFloat(4, 5f, -1f)
            };
            var points3D = new List<IPointIdFloat>() {
                new PointIdFloat(5, 1f, 2, 55f),
                new PointIdFloat(6, 0f, 1, 0.1f),
                new PointIdFloat(7, 3f, 3, 0.01f),
                new PointIdFloat(8, 1.5f, 2, -0.03f),
                new PointIdFloat(9, 5f, -1, 44f),
                new PointIdFloat(10, 15f, -51f, 73f),
                new PointIdFloat(11, 0.5f, -21f, 144f)
            };

            //should return the only point for a singleton tree
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { point2D });

            Assert.AreEqual(point2D, kdt.NearestNeighbour(point2D));
            Assert.AreEqual(point2D, kdt.NearestNeighbour(new PointIdFloat(156, 1f, 2f)));

            //should return the closest point (2D)
            kdt = new KdTree(points2D);
            Assert.IsTrue(points2D.All(p => kdt.NearestNeighbour(p) == p));

            Assert.AreEqual(points2D[0], kdt.NearestNeighbour(new PointIdFloat(12, 1.1f, 1.9f)));
            Assert.AreEqual(points2D[1], kdt.NearestNeighbour(new PointIdFloat(13, -5f, -5f)));
            Assert.AreEqual(points2D[2], kdt.NearestNeighbour(new PointIdFloat(14, 3.01f, 2.9999f)));
            Assert.AreEqual(points2D[3], kdt.NearestNeighbour(new PointIdFloat(15, 1.6f, 2f)));
            Assert.AreEqual(points2D[4], kdt.NearestNeighbour(new PointIdFloat(16, 160f, 2f)));

            //should return the closest point (3D)
            kdt = new KdTree(points3D);
            Assert.IsTrue(points3D.All(p => kdt.NearestNeighbour(p) == p));

            Assert.AreEqual(points3D[0], kdt.NearestNeighbour(new PointIdFloat(17, 1.1f, 1.9f, 49f)));
            Assert.AreEqual(points3D[1], kdt.NearestNeighbour(new PointIdFloat(18, -5f, -5f, 1f)));
            Assert.AreEqual(points3D[2], kdt.NearestNeighbour(new PointIdFloat(19, 3.01f, 2.9999f, -1f)));
            Assert.AreEqual(points3D[2], kdt.NearestNeighbour(new PointIdFloat(20, 160f, 2f, 0.1f)));
            Assert.AreEqual(points3D[3], kdt.NearestNeighbour(new PointIdFloat(21, 1.6f, 2f, 0f)));
            Assert.AreEqual(points3D[4], kdt.NearestNeighbour(new PointIdFloat(22, 160f, 2f, 40f)));
        }

        #endregion

        #region NearestNNeighbours

        [TestMethod()]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClasNearestNNeighboursNullPointTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f) });
            kdt.NearestNNeighbours(null, 1);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Wrong Point dimension: expected 2, got 3")]
        public void ClasNearestNNeighboursWrongDimensionPointTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f) });
            kdt.NearestNNeighbours(new PointIdFloat(1, 1f, 2f, 3f), 2);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Argument 'N': passed 0 while it must be positive")]
        public void ClasNearestNNeighboursInvalidSizeZeroTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f) });
            kdt.NearestNNeighbours(new PointIdFloat(1, 2f, 3f), 0);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Argument 'N': passed -3 while it must be positive")]
        public void ClasNearestNNeighboursInvalidSizeNegativeTest()
        {
            KdTree kdt = new KdTree(new List<IPointIdFloat>() { new PointIdFloat(0, 1f, 2f, 3f) });
            kdt.NearestNNeighbours(new PointIdFloat(1, -2f, -3f, -4f), -3);
        }

        [TestMethod()]
        public void ClasNearestNNeighboursTest()
        {
            var points2D = new List<IPointIdFloat>() {
                new PointIdFloat(0, 1f, 2f),
                new PointIdFloat(1, 0f, 1f),
                new PointIdFloat(2, 3f, 3f),
                new PointIdFloat(3, 1.5f, 2f),
                new PointIdFloat(4, 5f, -1f)
            };
            var points3D = new List<IPointIdFloat>() {
                new PointIdFloat(5, 1f, 2f, 55f),
                new PointIdFloat(7, 0f, 1f, 0.1f),
                new PointIdFloat(8, 3f, 3f, 0.01f),
                new PointIdFloat(9, 1.5f, 2f, -0.03f),
                new PointIdFloat(12, 5f, -1f, 44f),
                new PointIdFloat(13, 15f, -51f, 73f),
                new PointIdFloat(14, 0.5f, -21f, 144f)
            };

            //should return the closest N points (2D)
            var kdt = new KdTree(points2D);
            Assert.IsTrue(points2D.All(p =>
                kdt.NearestNNeighbours(p, 1).SequenceEqual(new List<IPointIdFloat>() { p })));

            var expectedResult = new List<IPointIdFloat>() {
                points2D[0]
            };
            var result = kdt.NearestNNeighbours(new PointIdFloat(0, 1.1f, 1.9f), 1).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points2D[1],
                points2D[0]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(1, -5f, -5f), 2).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points2D[2]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(2, 3.01f, 2.9999f), 1).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points2D[3],
                points2D[0],
                points2D[2]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(3, 1.6f, 2f), 3).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points2D[3],
                points2D[0],
                points2D[2],
                points2D[1]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(4, 1.6f, 2f), 4).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points2D[4],
                points2D[2]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(6, 160f, 2f), 2).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            //should return the closest points (3D)
            kdt = new KdTree(points3D);
            Assert.IsTrue(points3D.All(p =>
                kdt.NearestNNeighbours(p, 1).SequenceEqual(new List<IPointIdFloat>() { p })));

            expectedResult = new List<IPointIdFloat>() {
                points3D[0]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(7, 1.1f, 1.9f, 49f), 1).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points3D[1]
            };
            result = kdt.NearestNNeighbours(new PointIdFloat(8, -5f, -5f, 1f), 1).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));

            expectedResult = new List<IPointIdFloat>() {
                points3D[2],
                points3D[3],
                points3D[1],

            };
            result = kdt.NearestNNeighbours(new PointIdFloat(10, 3.01f, 2.9999f, -1f), 3).ToList();
            Assert.IsTrue(expectedResult.SequenceEqual(result));
        }

        #endregion

        #region [Internal] Median

        [TestMethod()]
        public void ClasMedianTest()
        {
            IPointIdFloat p0 = new PointIdFloat(0, 0f);
            IPointIdFloat p1 = new PointIdFloat(1, 1f);
            IPointIdFloat p2 = new PointIdFloat(2, 2f);
            IPointIdFloat p3 = new PointIdFloat(3, 3f);
            IPointIdFloat p4 = new PointIdFloat(4, 4f);

            IPointIdFloat p2D0 = new PointIdFloat(5, 0f, 2f);
            IPointIdFloat p2D1 = new PointIdFloat(6, 1f, 3f);
            IPointIdFloat p2D2 = new PointIdFloat(7, 2f, 4f);
            IPointIdFloat p2D3 = new PointIdFloat(8, 3f, 1f);
            IPointIdFloat p2D4 = new PointIdFloat(0, 4f, 0f);

            testMedian(new List<IPointIdFloat>() { p1 }, p1, new List<IPointIdFloat>(), new List<IPointIdFloat>());
            testMedian(new List<IPointIdFloat>() { p2, p1 }, p1, new List<IPointIdFloat>(), new List<IPointIdFloat>() { p2 });
            testMedian(new List<IPointIdFloat>() { p1, p2 }, p1, new List<IPointIdFloat>(), new List<IPointIdFloat>() { p2 });
            testMedian(new List<IPointIdFloat>() { p1, p2, p0 }, p1, new List<IPointIdFloat>() { p0 }, new List<IPointIdFloat>() { p2 });
            testMedian(new List<IPointIdFloat>() { p2, p1, p0, p1, p1 }, p1, new List<IPointIdFloat>() { p0, p1, p1 }, new List<IPointIdFloat>() { p2 });
            testMedian(new List<IPointIdFloat>() { p2D2, p2D1, p2D0, p2D3, p2D4 }, p2D2, new List<IPointIdFloat>() { p2D1, p2D0 }, new List<IPointIdFloat>() { p2D3, p2D4 });
            testMedian(new List<IPointIdFloat>() { p2D2, p2D1, p2D0, p2D3, p2D4 }, p2D0, new List<IPointIdFloat>() { p2D3, p2D4 }, new List<IPointIdFloat>() { p2D1, p2D2 }, 1);
        }

        #endregion

        #region Private

        private void testMedian(
            IList<IPointIdFloat> points,
            IPointIdFloat expectedMedian,
            IList<IPointIdFloat> expectedLeft,
            IList<IPointIdFloat> expectedRight,
            int depth = 0)
        {
            IList<IPointIdFloat> left = null;
            IList<IPointIdFloat> right = null;
            Func<IPointIdFloat, float> keySelector = t => KdTree.KdTreeNode.KeyByDepth(t, depth);

            IPointIdFloat median = KdTree.MedianPoint(points, ref left, ref right, depth, rnd);
            Assert.AreEqual(expectedMedian, median);
            Assert.IsTrue(SequenceEquivalent(expectedLeft, left, keySelector));
            Assert.IsTrue(SequenceEquivalent(expectedRight, right, keySelector));
        }

        private bool SequenceEquivalent<T, R>(IList<T> list1, IList<T> list2, Func<T, R> keySelector)
        {
            return list1.OrderBy(keySelector).SequenceEqual(list2.OrderBy(keySelector));
        }

        #endregion
    }
}