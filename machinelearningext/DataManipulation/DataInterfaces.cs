// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Ext.DataManipulation
{
    public interface IDataFrameView
    {
    }

    public interface IDataContainer
    {
        IDataColumn GetColumn(int col);
    }

    public interface IDataColumn
    {
        object Get(int row);
        void Set(int row, object value);
    }
}
