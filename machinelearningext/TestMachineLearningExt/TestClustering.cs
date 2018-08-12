// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Scikit.ML.Clustering;
using Scikit.ML.NearestNeighbors;
using Scikit.ML.TestHelper;
using Scikit.ML.PipelineHelper;


namespace TestMachineLearningExt
{
    [TestClass]
    public class TestClustering
    {
        private static readonly Random rnd = new Random();

        #region DbScan

        [TestMethod()]
        public void StartTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);

            var dbscan = new DBScan(points);
            var result = dbscan.Cluster(0.5f, 2);
            Assert.IsTrue(result.Count == points.Count);
            Assert.IsTrue(result.Values.Max() <= points.Count);
        }

        [TestMethod()]
        public void RegionQueryPointsListTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);
            var p = new PointIdFloat(new List<float>() { 1.5f, 1.5f, 1.5f });
            Func<IPointIdFloat, Tuple<float, float, float>> keySelector = t => Tuple.Create(
                KdTree.KdTreeNode.KeyByDepth(t, 0), 0.0f, 0.0f);

            var result = DBScan.RegionQuery(points, p, 0.16f);

            Assert.IsTrue(result.Count() == 0);

            result = DBScan.RegionQuery(points, p, 1);
            var expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));

            p = new PointIdFloat(new List<float>() { 0.75f, 0.75f, 0.75f });
            result = DBScan.RegionQuery(points, p, (float)Math.Sqrt(2f / 3f));
            expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));
        }

        [TestMethod()]
        public void RegionQueryKdTreeTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);

            var p = new PointIdFloat(new List<float>() { 1.5f, 1.5f, 1.5f });
            KdTree kdt = new KdTree(points);

            var result = kdt.PointsWithinDistance(p, 0.5f);

            Assert.IsTrue(result.Count == 0);

            result = DBScan.RegionQuery(kdt, p, 1);
            var expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));

            p = new PointIdFloat(new List<float>() { 0.75f, 0.75f, 0.75f });
            result = kdt.PointsWithinDistance(p, (float)Math.Sqrt(2));
            expectedResult = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
            };
            PointIdFloat.SetIds(expectedResult);

            Assert.IsTrue(SequenceEquivalent(result.ToList(), expectedResult, PointIdFloat.PointsComparison));
        }

        [TestMethod()]
        public void RegionQueryTest()
        {
            //Checks that both RegionQuery overloaded methods always return the same result
            for (int d = 2; d <= 5; d++)
            {
                var points = new List<IPointIdFloat>();
                for (int i = 0; i < 200; i++)
                    points.Add(RandomPoint(d));
                KdTree kdt = new KdTree(points);

                for (int i = 0; i < 50; i++)
                {
                    var p = RandomPoint(d);
                    float epsilon = (float)rnd.NextDouble() * int.MaxValue;
                    var l1 = DBScan.RegionQuery(points, p, epsilon).ToList();
                    var l2 = DBScan.RegionQuery(kdt, p, epsilon).ToList();
                    var b = SequenceEquivalent(l1, l2, PointIdFloat.PointsComparison);
                    if (!b)
                    {
                        var l3 = DBScan.RegionQuery(kdt, p, epsilon).ToList();
                        Assert.IsTrue(SequenceEquivalent(l3, l2, PointIdFloat.PointsComparison));
                        Assert.IsTrue(false);
                    }
                }
            }
        }

        [TestMethod()]
        public void ExpandRegionTest()
        {
            var points = new List<IPointIdFloat>()
            {
                new PointIdFloat(new List<float>() { 1.5f, 1.5f, 1.5f }),
                new PointIdFloat(new List<float>() { 1, 1, 1 }),
                new PointIdFloat(new List<float>() { 2, 1, 1 }),
                new PointIdFloat(new List<float>() { 1, 2, 1 }),
                new PointIdFloat(new List<float>() { 1, 1, 2 }),
                new PointIdFloat(new List<float>() { 3, 3, 3 }),
                new PointIdFloat(new List<float>() { 2, 2, 2 })
            };
            PointIdFloat.SetIds(points);

            var clusters = new Dictionary<long, int>();
            var visited = new HashSet<long>();
            var p = points[0];
            float epsilon = 1f;

            var neighbours = DBScan.RegionQuery(points, p, epsilon);
            Assert.IsTrue(neighbours.Contains(p));

            DBScan.ExpandCluster(clusters, visited, points, p, neighbours.ToList(), 0, epsilon, 3);
            Assert.IsTrue(clusters.ContainsKey(p.id));
            foreach (var q in neighbours)
            {
                Assert.IsTrue(clusters.ContainsKey(q.id));
                Assert.IsTrue(clusters[q.id] == clusters[p.id]);
            }

            Assert.IsFalse(clusters.ContainsKey(points.Count() + 1));
        }

        private bool SequenceEquivalent<T>(List<T> list1, List<T> list2, Comparison<T> comparer)
        {
            list1.Sort(comparer);
            list2.Sort(comparer);
            return list1.SequenceEqual(list2);
        }

        private IPointIdFloat RandomPoint(int dimension = 2)
        {
            var coordinates = new List<float>();
            for (int i = 0; i < dimension; i++)
                coordinates.Add((float)(1000 * rnd.NextDouble()));
            return new PointIdFloat(coordinates);
        }

        [TestMethod]
        public void TestDBScanTransform()
        {
            var methodName = System.Reflection.MethodBase.GetCurrentMethod().Name;
            var dataFilePath = FileHelper.GetTestFile("three_classes_2d.txt");
            var outputDataFilePath = FileHelper.GetOutputFile("outputDataFilePath.txt", methodName);
            var outModelFilePath = FileHelper.GetOutputFile("outModelFilePath.zip", methodName);

            var env = EnvHelper.NewTestEnvironment();
            var loader = env.CreateLoader("text{col=RowId:I4:0 col=Features:R4:1-2}", new MultiFileSource(dataFilePath));
            var xf = env.CreateTransform("DBScan{col=Features}", loader);

            string schema = SchemaHelper.ToString(xf.Schema);
            if (string.IsNullOrEmpty(schema))
                throw new Exception("Schema is null.");
            if (!schema.Contains("Cluster"))
                throw new Exception("Schema does not contain Cluster.");
            if (!schema.Contains("Score"))
                throw new Exception("Schema does not contain Score.");

            StreamHelper.SaveModel(env, xf, outModelFilePath);

            var saver = env.CreateSaver("Text{header=- schema=-}");
            using (var fs2 = File.Create(outputDataFilePath))
                saver.SaveData(fs2, TestTransformHelper.AddFlatteningTransform(env, xf),
                                        StreamHelper.GetColumnsIndex(xf.Schema, new[] { "Features", "ClusterId", "Score" }));

            // Checking the values.
            var lines = File.ReadAllLines(outputDataFilePath).Select(c => c.Split('\t')).Where(c => c.Length == 4);
            if (!lines.Any())
                throw new Exception(string.Format("The output file is empty or not containing three columns '{0}'", outputDataFilePath));
            var clusters = lines.Select(c => c[1]).Distinct();
            if (clusters.Count() <= 1)
                throw new Exception("Only one cluster, this is unexpected.");

            // Serialization.
            var outData = FileHelper.GetOutputFile("outData1.txt", methodName);
            var outData2 = FileHelper.GetOutputFile("outData2.txt", methodName);
            TestTransformHelper.SerializationTestTransform(env, outModelFilePath, xf, loader, outData, outData2);
        }

        #endregion
    }
}
