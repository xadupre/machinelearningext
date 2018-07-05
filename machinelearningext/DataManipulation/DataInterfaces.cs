// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Interface for a data container held by a dataframe.
    /// </summary>
    public interface IDataContainer
    {
        /// <summary>
        /// Returns a columns based on its position.
        /// </summary>
        IDataColumn GetColumn(int col);

        /// <summary>
        /// Orders the rows.
        /// </summary>
        void Order(int[] order);
    }

    public delegate void GetterAt<DType>(int i, ref DType value);

    /// <summary>
    /// Interface for a column container.
    /// </summary>
    public interface IDataColumn
    {
        /// <summary>
        /// Length of the column
        /// </summary>
        int Length { get; }

        /// <summary>
        /// type of the column 
        /// </summary>
        DataKind Kind { get; }

        /// <summary>
        /// Returns a copy.
        /// </summary>
        IDataColumn Copy();

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        IDataColumn Copy(IEnumerable<int> rows);

        /// <summary>
        /// Returns the element at position row
        /// </summary>
        object Get(int row);

        /// <summary>
        /// Get a getter for a specific location.
        /// </summary>
        GetterAt<DType> GetGetterAt<DType>()
            where DType : IEquatable<DType>, IComparable<DType>;

        /// <summary>
        /// Updates value at position row
        /// </summary>
        void Set(int row, object value);

        /// <summary>
        /// Updates all values.
        /// </summary>
        void Set(object value);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<bool> rows, object value);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<int> rows, object value);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<bool> rows, IEnumerable<object> values);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<int> rows, IEnumerable<object> values);

        /// <summary>
        /// The returned getter returns the element
        /// at position <pre>cursor.Position</pre>
        /// </summary>
        ValueGetter<DType> GetGetter<DType>(IRowCursor cursor);

        /// <summary>
        /// exact comparison
        /// </summary>
        bool Equals(IDataColumn col);

        /// <summary>
        /// Returns an enumerator on every row telling if each of them
        /// verfies the condition.
        /// </summary>
        IEnumerable<bool> Filter<TSource>(Func<TSource, bool> predicate);

        /// <summary>
        /// Applies the same function on every value of the column. Example:
        /// <code>
        /// var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
        /// var df = DataFrame.ReadStr(text);
        /// df["fAA"] = df["AA"].Apply((ref DvInt4 vin, ref float vout) => { vout = (float)vin; });
        /// </code>
        /// </summary>
        NumericColumn Apply<TSrc, TDst>(ValueMapper<TSrc, TDst> mapper)
            where TDst : IEquatable<TDst>, IComparable<TDst>;

        /// <summary>
        /// Sorts the column. Returns the order 
        /// </summary>
        void Sort(ref int[] order, bool ascending = true);
        int[] Sort(bool ascending = true, bool inplace = true);

        /// <summary>
        /// Orders the rows.
        /// </summary>
        void Order(int[] order);
    }

    /// <summary>
    /// Interface for dataframes and dataframe views.
    /// </summary>
    public interface IDataFrameView : IDataView, IEquatable<IDataFrameView>
    {
        /// <summary>
        /// All rows or all columns.
        /// </summary>
        int[] ALL { get; }

        /// <summary>
        /// Returns the number of rows.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Returns the number of columns.
        /// </summary>
        int ColumnCount { get; }

        /// <summary>
        /// Returns the shape of the dataframe (number of rows, number of columns).
        /// </summary>
        Tuple<int, int> Shape { get; }

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        DataFrame Copy();

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        DataFrame Copy(IEnumerable<int> rows, IEnumerable<int> columns);

        /// <summary>
        /// Sames a GetRowCursor but on a subset of the data.
        /// </summary>
        IRowCursor GetRowCursor(int[] rows, int[] columns, Func<int, bool> needCol, IRandom rand = null);

        /// <summary>
        /// Sames a GetRowCursorSet but on a subset of the data.
        /// </summary>
        IRowCursor[] GetRowCursorSet(int[] rows, int[] columns, out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null);

        /// <summary>
        /// Retrieves a column by its name.
        /// </summary>
        NumericColumn GetColumn(string colname, int[] rows = null);

        /// <summary>
        /// Retrieves a column by its position.
        /// </summary>
        NumericColumn GetColumn(int col, int[] rows = null);

        /// <summary>
        /// Drops some columns.
        /// Data is not copied.
        /// </summary>
        DataFrameView Drop(IEnumerable<string> colNames);

        /// <summary>
        /// Orders the rows.
        /// </summary>
        void Order(int[] order);

        /// <summary>
        /// Enumerates tuples of MutableTuple.
        /// The iterated items are reused.
        /// </summary>
        /// <typeparam name="TTuple">item type</typeparam>
        /// <param name="columns">list of columns to select</param>
        /// <param name="ascending">order</param>
        /// <param name="rows">subset of rows</param>
        /// <returns>enumerator on MutableTuple</returns>
        IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>;
        IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;


        /// <summary>
        /// Sorts by rows.
        /// </summary>
        void Sort<T1>(IEnumerable<string> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>;

        /// <summary>
        /// Sorts by rows.
        /// </summary>
        void Sort<T1, T2>(IEnumerable<string> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;

        /// <summary>
        /// Sorts by rows.
        /// </summary>
        void Sort<T1, T2, T3>(IEnumerable<string> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;
    }
}
