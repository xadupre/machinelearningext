// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Interface for dataframes and dataframe views.
    /// </summary>
    public interface IDataFrameView
    {
    }

    /// <summary>
    /// Interface for a data container held by a dataframe.
    /// </summary>
    public interface IDataContainer
    {
        IDataColumn GetColumn(int col);
    }

    /// <summary>
    /// Interface for a column container.
    /// </summary>
    public interface IDataColumn
    {
        object Get(int row);
        void Set(int row, object value);
    }
}
