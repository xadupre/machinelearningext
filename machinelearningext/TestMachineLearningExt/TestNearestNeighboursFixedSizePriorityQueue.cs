// See the LICENSE file in the project root for more information.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Ext.NearestNeighbors;


namespace TestMachineLearningExt
{
    [TestClass()]
    public class TestNearestNeighborsFixedSizePriorityQueue
    {
        private static readonly Random Rnd = new Random();
        private static readonly object syncLock = new object();

        private static readonly IList<KeyValuePair<float, int>> Elements = new List<KeyValuePair<float, int>>()
            {
                new KeyValuePair<float, int>(0, 1),
                new KeyValuePair<float, int>(10, 11),
                new KeyValuePair<float, int>(20, 21)
            };

        [TestMethod()]
        public void ClasFixedSizePriorityQueueConstructorTest()
        {
            FixedSizePriorityQueue<float, int> q = new FixedSizePriorityQueue<float, int>(1);
            Assert.IsNotNull(q);
            Assert.IsTrue(q.IsEmpty);
            Assert.IsFalse(q.IsFull);

            q = new FixedSizePriorityQueue<float, int>(Elements.Take(1), 2);
            Assert.IsNotNull(q);
            Assert.IsFalse(q.IsEmpty);
            Assert.IsFalse(q.IsFull);

            q = new FixedSizePriorityQueue<float, int>(Elements.Take(2), 2);
            Assert.IsNotNull(q);
            Assert.IsFalse(q.IsEmpty);
            Assert.IsTrue(q.IsFull);

            q = new FixedSizePriorityQueue<float, int>(Elements.Take(3), 3);
            Assert.IsNotNull(q);
            Assert.IsFalse(q.IsEmpty);
            Assert.IsTrue(q.IsFull);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Argument 'size' must be positive")]
        public void ClasFixedSizePriorityQueueConstructorFailSizeZeroTest()
        {
            new FixedSizePriorityQueue<float, int>(0);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Argument 'size' must be positive")]
        public void ClasFixedSizePriorityQueueConstructorFailSizeNegativeTest()
        {
            new FixedSizePriorityQueue<float, int>(-2);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Argument 'size' must be positive")]
        public void ClasFixedSizePriorityQueueConstructor2FailSizeNegativeTest()
        {
            new FixedSizePriorityQueue<float, int>(Elements.Take(1), -2);
        }

        [TestMethod()]
        [ExpectedException(typeof(ArgumentException), "Queue size too small for the init collection")]
        public void ClasFixedSizePriorityQueueConstructorFailWhenCollectionBiggerThanSizeTest()
        {
            new FixedSizePriorityQueue<float, int>(Elements, 2);
        }

        [TestMethod()]
        public void ClasPeekTest()
        {
            FixedSizePriorityQueue<float, int> testQ = new FixedSizePriorityQueue<float, int>(10);
            Assert.IsNull(testQ.Peek());

            testQ = new FixedSizePriorityQueue<float, int>(
                new List<KeyValuePair<float, int>>() {
                    new KeyValuePair<float, int>(-10, 22)
                },
                10);
            Assert.AreEqual(1, testQ.Count);
            Assert.IsNotNull(testQ.Peek());
            Assert.AreEqual(1, testQ.Count);
            Assert.AreEqual(-10, testQ.Peek().Value.Key);
            Assert.AreEqual(22, testQ.Peek().Value.Value);
        }

        [TestMethod()]
        public void ClasEnqueueTest()
        {
            int maxSize = 3;

            FixedSizePriorityQueue<float, int> testQ = new FixedSizePriorityQueue<float, int>(maxSize);
            Assert.IsNull(testQ.Peek());
            Assert.AreEqual(0, testQ.Count);

            float key = -5;
            int value = 1;

            testQ.Enqueue(key, value);

            Assert.AreEqual(1, testQ.Count);
            Assert.AreEqual(key, testQ.Peek().Value.Key);
            Assert.AreEqual(value, testQ.Peek().Value.Value);

            key = 10;
            value = 2;

            testQ.Enqueue(key, value);

            Assert.AreEqual(2, testQ.Count);
            Assert.AreEqual(-5, testQ.Peek().Value.Key);
            Assert.AreEqual(1, testQ.Peek().Value.Value);

            //Updates the top of the queue when higher prio elements are added
            key = -10;
            value = 3;

            testQ.Enqueue(key, value);

            Assert.AreEqual(3, testQ.Count);
            Assert.AreEqual(key, testQ.Peek().Value.Key);
            Assert.AreEqual(value, testQ.Peek().Value.Value);

            //When the queue is full, doesn't change its size
            //and doesn't insert a new element with lower prio than top's
            key = -11;
            value = 4;

            testQ.Enqueue(key, value);

            Assert.AreEqual(3, testQ.Count);
            Assert.AreEqual(-10, testQ.Peek().Value.Key);
            Assert.AreEqual(3, testQ.Peek().Value.Value);

            //When the queue is full, doesn't change its size
            //and does insert a new element with hiher prio than top's
            key = -9;
            value = 5;

            testQ.Enqueue(key, value);

            Assert.AreEqual(3, testQ.Count);
            Assert.AreEqual(key, testQ.Peek().Value.Key);
            Assert.AreEqual(value, testQ.Peek().Value.Value);
        }

        [TestMethod()]
        public void ClasShouldOnlyStoreNLArgestValuesTest()
        {
            foreach (int size in new List<int>() { 10, 100, 150 })
            {
                FixedSizePriorityQueue<int, bool> testQ = new FixedSizePriorityQueue<int, bool>(size);
                int n = 4 * size;
                List<int> keys = new List<int>(n);
                for (int i = 0; i < n; i++)
                {
                    int key;
                    lock (syncLock)
                    {
                        key = Rnd.Next();
                    }

                    keys.Add(key);
                    testQ.Enqueue(key, true);
                }
                keys.Sort();
                List<int> largestKeys = keys.Skip(n - size).ToList();
                Assert.AreEqual(size, largestKeys.Count);

                var result = testQ.ToList()
                    .Select(kvp => kvp.Key)
                    .ToList();
                result.Sort();
                Assert.IsTrue(largestKeys.SequenceEqual(result));
            }
        }
    }
}
