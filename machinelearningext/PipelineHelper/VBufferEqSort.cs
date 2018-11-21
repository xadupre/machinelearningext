// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Scikit.ML.PipelineHelper
{
    public struct VBufferEqSort<T> : IEquatable<VBufferEqSort<T>>, IComparable<VBufferEqSort<T>>
        where T : IEquatable<T>, IComparable<T>
    {
        public VBuffer<T> data;

        #region VBuffer API

        public int Length => data.Length;
        public int Count => data.Count;
        public T[] Values => data.Values;
        public int[] Indices => data.Indices;

        public VBufferEqSort(VBuffer<T> v) { data = new VBuffer<T>(); v.CopyTo(ref data); }
        public VBufferEqSort(int length, T[] values, int[] indices = null) { data = new VBuffer<T>(length, values, indices); }
        public VBufferEqSort(int length, int count, T[] values, int[] indices) { data = new VBuffer<T>(length, count, values, indices); }

        public bool IsDense => data.IsDense;

        public static void Copy(T[] src, int srcIndex, ref VBuffer<T> dst, int length) { VBuffer<T>.Copy(src, srcIndex, ref dst, length); }
        public static void Copy(T[] src, int srcIndex, ref VBufferEqSort<T> dst, int length) { VBuffer<T>.Copy(src, srcIndex, ref dst.data, length); }
        public static void Copy(ref VBuffer<T> src, ref VBuffer<T> dst) { src.CopyTo(ref dst); }
        public static void Copy(ref VBufferEqSort<T> src, ref VBuffer<T> dst) { src.data.CopyTo(ref dst); }
        public static void Copy(ref VBuffer<T> src, ref VBufferEqSort<T> dst) { src.CopyTo(ref dst.data); }
        public static void Copy(ref VBufferEqSort<T> src, ref VBufferEqSort<T> dst) { src.data.CopyTo(ref dst.data); }

        public void CopyTo(ref VBuffer<T> dst) { data.CopyTo(ref dst); }
        public void CopyTo(ref VBuffer<T> dst, int srcMin, int length) { data.CopyTo(ref dst, srcMin, length); }

        public void CopyTo(T[] dst) { data.CopyTo(dst); }
        public void CopyTo(T[] dst, int ivDst, T defaultValue = default(T)) { data.CopyTo(dst, ivDst, defaultValue); }
        //public void CopyTo(ref VBuffer<T> dst, int[] indicesInclude, int count) { data.CopyTo(ref dst, indicesInclude, count); }
        public void CopyToDense(ref VBuffer<T> dst) { data.CopyToDense(ref dst); }
        public IEnumerable<T> DenseValues() { return data.DenseValues(); }
        public void GetItemOrDefault(int slot, ref T dst) { data.GetItemOrDefault(slot, ref dst); }
        public IEnumerable<KeyValuePair<int, T>> Items(bool all = false) { return data.Items(all); }

        public T GetItemOrDefault(int slot)
        {
            return slot < Length ? data.GetItemOrDefault(slot) : default(T);
        }

        public IEnumerable<KeyValuePair<int, T>> SparseValues()
        {
            if (IsDense)
            {
                for (int i = 0; i < Length; ++i)
                    yield return new KeyValuePair<int, T>(i, Values[i]);
            }
            else
            {
                for (int i = 0; i < Length; ++i)
                    yield return new KeyValuePair<int, T>(Indices[i], Values[i]);
            }
        }


        #endregion

        #region IEquatable, IComparable

        public bool Equals(VBufferEqSort<T> other)
        {
            if (Length != other.Length)
                return false;
            if (Count != other.Count)
                return false;
            int vi, vj;
            for (int i = 0; i < Count; ++i)
            {
                if (!Values[i].Equals(other.Values[i]))
                    return false;
                vi = data.Indices == null ? i : data.Indices[i];
                vj = other.data.Indices == null ? i : other.data.Indices[i];
                if (vi != vj)
                    return false;
            }
            return true;
        }

        public int CompareTo(VBufferEqSort<T> other)
        {
            var enum1 = SparseValues().GetEnumerator();
            var enum2 = other.SparseValues().GetEnumerator();
            int r;
            while (true)
            {
                if (!enum2.MoveNext())
                    return enum1.MoveNext() ? -1 : 0;
                if (!enum1.MoveNext())
                    return 1;
                if (enum1.Current.Key == enum2.Current.Key)
                {
                    r = enum1.Current.Value.CompareTo(enum2.Current.Value);
                    if (r != 0)
                        return r;
                }
                else
                    return enum1.Current.Key > enum2.Current.Key ? -1 : 1;
            }
        }

        #endregion
    }
}
