// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;


namespace Microsoft.ML.Ext.NearestNeighbours
{
    /// <summary>
    /// Stores at most #size elements in a priority queue.
    /// Each element is an item with a key, which determines its priority, and a value
    /// associated to it.
    /// If more than #size elements are added, only the largest #size are kept.
    /// </summary>
    /// <typeparam name="TKey">Type for the keys (which will determine element's priority)</typeparam>
    /// <typeparam name="TValue">Type for the values</typeparam>
    public class FixedSizePriorityQueue<TKey, TValue> :
        PriorityQueue<TKey, TValue>
        where TKey : IComparable<TKey>
    {
        private readonly int size;

        /// <summary>
        /// Initializes a new instance of the ConcurrentPriorityQueue class.
        /// </summary>
        public FixedSizePriorityQueue(int size) : base()
        {
            ValidateSize(size);
            this.size = size;
        }

        /// <summary>
        /// Initializes a new instance of the ConcurrentPriorityQueue class that
        /// contains elements copied from the specified collection.
        /// If the collection is bigger than the max size of the queue, an exception is thrown.
        /// This is because
        /// 1) It could be confusing just discarding elements without notifying the caller.
        /// 2) It would require adding elements one by one, preventing from using efficient
        ///    O(n) construction for the heap.
        /// </summary>
        /// <param name="collection">The collection whose elements are copied to the new ConcurrentPriorityQueue.</param>
        /// <param name="size">size</param>
        public FixedSizePriorityQueue(
            IEnumerable<KeyValuePair<TKey, TValue>> collection,
            int size) : base(collection)
        {
            ValidateSize(size);
            this.size = size;
            if (this.Count > size)
            {
                throw new ArgumentException("Queue size too small for the init collection");
            }
        }

        public bool IsFull
        {
            get
            {
                return Count == size;
            }
        }

        public KeyValuePair<TKey, TValue>? Peek()
        {
            if (IsEmpty)
                return null;
            else
            {
                KeyValuePair<TKey, TValue> tmp;
                base.TryPeek(out tmp);
                return tmp;
            }
        }

        /// <summary>
        /// Adds the key/value pair to the priority queue.
        /// If the queue is already full, then removes and return the lowest priority element,
        /// before inserting the new one - but if and only if the priority of the new element
        /// is not lower than the priority of the old one.
        /// </summary>
        /// <param name="priority">The priority of the item to be added.</param>
        /// <param name="value">The item to be added.</param>
        /// <returns>The element removed from the queue, if any, or null.</returns>
        public new KeyValuePair<TKey, TValue>? Enqueue(TKey priority, TValue value)
        {
            return Enqueue(new KeyValuePair<TKey, TValue>(priority, value));
        }

        /// <summary>
        /// Adds the key/value pair to the priority queue.
        /// If the queue is already full, then removes and return the lowest priority element,
        /// before inserting the new one - but if and only if the priority of the new element
        /// is not lower than the priority of the old one.
        /// </summary>
        /// <param name="item">The key/value pair to be added to the queue.</param>
        /// <returns>The element removed from the queue, if any, or null.</returns>
        public new KeyValuePair<TKey, TValue>? Enqueue(KeyValuePair<TKey, TValue> item)
        {
            if (IsEmpty || this.Count < size)
            {
                base.Enqueue(item);
                return null;
            }
            else if (Peek().Value.Key.CompareTo(item.Key) < 0)
            {
                KeyValuePair<TKey, TValue> tmp;
                TryDequeue(out tmp);
                base.Enqueue(item);
                return tmp;
            }
            return null;
        }

        private void ValidateSize(int size)
        {
            if (size <= 0)
                throw new ArgumentException("Argument 'size' must be positive.");
        }
    }
}