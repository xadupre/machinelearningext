// See the LICENSE file in the project root for more information.

using System.Collections.Immutable;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Scikit.ML.PipelineHelper
{
    /// <summary>
    /// Implements functions declared as internal.
    /// </summary>
    public static class ColumnTypeHelper
    {
        public static bool IsPrimitive(this ColumnType column)
        {
            return column is PrimitiveType;
        }

        public static bool IsVector(this ColumnType column)
        {
            return column is VectorType;
        }

        public static bool IsNumber(this ColumnType column)
        {
            return column is NumberType;
        }

        public static bool IsKey(this ColumnType column)
        {
            return column is KeyType;
        }

        public static bool IsText(this ColumnType column)
        {
            if (!(column is TextType))
                return false;
            Contracts.Assert(column == TextType.Instance);
            return true;
        }

        public static KeyType AsKey(this ColumnType column)
        {
            return IsKey(column) ? (KeyType)column : null;
        }

        public static PrimitiveType AsPrimitive(this ColumnType column)
        {
            return IsPrimitive(column) ? (PrimitiveType)column : null;
        }

        public static VectorType AsVector(this ColumnType column)
        {
            return IsVector(column) ? (VectorType)column : null;
        }

        public static int KeyCount(this ColumnType column)
        {
            return IsKey(column) ? AsKey(column).Count : 0;
        }

        public static int DimCount(this ColumnType column)
        {
            return IsVector(column) ? AsVector(column).Dimensions.Length : 0;
        }

        public static int GetDim(this ColumnType column, int dim)
        {
            return IsVector(column) ? AsVector(column).Dimensions[dim] : 0;
        }

        public static int ValueCount(this ColumnType column)
        {
            return IsVector(column) ? AsVector(column).Size : 1;
        }

        public static PrimitiveType ItemType(this ColumnType column)
        {
            return IsVector(column) ? AsVector(column).ItemType : AsPrimitive(column);
        }

        public static DataKind RawKind(this ColumnType type)
        {
            DataKind kind;
            if (IsVector(type))
                if (DataKindExtensions.TryGetDataKind(ItemType(type).RawType, out kind))
                    return kind;
                else
                if (DataKindExtensions.TryGetDataKind(type.RawType, out kind))
                    return kind;
            throw Contracts.ExceptNotSupp($"Unable to guess kind for type {type}.");
        }

        public static bool IsKnownSizeVector(this ColumnType column)
        {
            return !IsVector(column) || (AsVector(column).Size > 0);
        }

        public static int VectorSize(this ColumnType column)
        {
            return IsVector(column) ? AsVector(column).Size : 0;
        }

        public static NumberType NumberFromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.I1: return NumberType.I1;
                case DataKind.U1: return NumberType.U1;
                case DataKind.I2: return NumberType.I2;
                case DataKind.U2: return NumberType.U2;
                case DataKind.I4: return NumberType.I4;
                case DataKind.U4: return NumberType.U4;
                case DataKind.I8: return NumberType.I8;
                case DataKind.U8: return NumberType.U8;
                case DataKind.R4: return NumberType.R4;
                case DataKind.R8: return NumberType.R8;
                case DataKind.UG: return NumberType.UG;
                default:
                    throw Contracts.ExceptNotImpl($"Number from kind not implemented for kind {kind}.");
            }
        }

        public static PrimitiveType PrimitiveFromKind(DataKind kind)
        {
            if (kind == DataKind.TX)
                return TextType.Instance;
            if (kind == DataKind.BL)
                return BoolType.Instance;
            if (kind == DataKind.TS)
                return TimeSpanType.Instance;
            if (kind == DataKind.DT)
                return DateTimeType.Instance;
            if (kind == DataKind.DZ)
                return DateTimeOffsetType.Instance;
            return NumberFromKind(kind);
        }

        public static bool SameSizeAndItemType(this ColumnType column, ColumnType other)
        {
            if (other == null)
                return false;

            if (column.Equals(other))
                return true;

            // For vector types, we don't care about the factoring of the dimensions.
            if (!IsVector(column) || !IsVector(other))
                return false;
            if (!ItemType(column).Equals(ItemType(other)))
                return false;
            return VectorSize(column) == VectorSize(other);
        }

        public static bool IsBool(this ColumnType column)
        {
            if (!(column is BoolType))
                return false;
            Contracts.Assert(column == BoolType.Instance);
            return true;
        }

        public static bool IsValidDataKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.U1:
                case DataKind.U2:
                case DataKind.U4:
                case DataKind.U8:
                    return true;
                default:
                    return false;
            }
        }
    }
}
