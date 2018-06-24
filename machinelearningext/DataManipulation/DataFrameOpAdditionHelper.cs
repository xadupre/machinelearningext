// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Ext.DataManipulation
{
    public static class DataFrameOperationHelper
    {
        #region Addition of a column and a value.

        static void Addition<T1, T3>(NumericColumn c1, out T1[] a, out DataColumn<T3> res)
            where T1 : IEquatable<T1>
            where T3 : IEquatable<T3>
        {
            var c1o = c1.Column as DataColumn<T1>;
            if (c1 == null)
                throw new DataTypeError(string.Format("Addition not implemented for type {0}.", c1.GetType()));
            res = new DataColumn<T3>(c1.Length);
            a = c1o.Data;
        }

        public static NumericColumn Addition(NumericColumn c1, DvInt4 value)
        {
            switch (c1.Kind)
            {
                case DataKind.I4:
                    {
                        DvInt4[] a;
                        DataColumn<DvInt4> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, a[i] + value);
                        return new NumericColumn(res);
                    }
                case DataKind.R4:
                    {
                        float[] a;
                        DataColumn<float> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, a[i] + (float)value);
                        return new NumericColumn(res);
                    }
                default:
                    throw new DataTypeError(string.Format("Addition not implemented for column {0}.", c1.Kind));
            }
        }

        public static NumericColumn Addition(NumericColumn c1, float value)
        {
            switch (c1.Kind)
            {
                case DataKind.I4:
                    {
                        DvInt4[] a;
                        DataColumn<float> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, (int)a[i] + value);
                        return new NumericColumn(res);
                    }
                case DataKind.R4:
                    {
                        float[] a;
                        DataColumn<float> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, a[i] + value);
                        return new NumericColumn(res);
                    }
                default:
                    throw new DataTypeError(string.Format("Addition not implemented for column {0}.", c1.Kind));
            }
        }

        public static NumericColumn Addition(NumericColumn c1, double value)
        {
            switch (c1.Kind)
            {
                case DataKind.I4:
                    {
                        DvInt4[] a;
                        DataColumn<double> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, (int)a[i] + value);
                        return new NumericColumn(res);
                    }
                case DataKind.R4:
                    {
                        float[] a;
                        DataColumn<double> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, a[i] + value);
                        return new NumericColumn(res);
                    }
                case DataKind.R8:
                    {
                        double[] a;
                        DataColumn<double> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, a[i] + value);
                        return new NumericColumn(res);
                    }
                default:
                    throw new DataTypeError(string.Format("Addition not implemented for column {0}.", c1.Kind));
            }
        }

        public static NumericColumn Addition(NumericColumn c1, DvText value)
        {
            switch (c1.Kind)
            {
                case DataKind.TX:
                    {
                        DvText[] a;
                        DataColumn<DvText> res;
                        Addition(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, new DvText(a[i].ToString() + value.ToString()));
                        return new NumericColumn(res);
                    }
                default:
                    throw new DataTypeError(string.Format("Addition not implemented for column {0}.", c1.Kind));
            }
        }

        #endregion

        #region Addition of two columns.

        static void Addition<T1, T2, T3>(NumericColumn c1, NumericColumn c2,
                                         out T1[] a, out T2[] b, out DataColumn<T3> res)
            where T1 : IEquatable<T1>
            where T2 : IEquatable<T2>
            where T3 : IEquatable<T3>
        {
            var c1o = c1.Column as DataColumn<T1>;
            var c2o = c2.Column as DataColumn<T2>;
            if (c1 == null || c2 == null)
                throw new DataTypeError(string.Format("Addition not implemented for {0}, {1}.", c1.Kind, c2.Kind));
            res = new DataColumn<T3>(c1.Length);
            a = c1o.Data;
            b = c2o.Data;
        }

        public static NumericColumn Addition(NumericColumn c1, NumericColumn c2)
        {
            switch (c1.Kind)
            {
                case DataKind.I4:
                    switch (c2.Kind)
                    {
                        case DataKind.I4:
                            {
                                DvInt4[] a;
                                DvInt4[] b;
                                DataColumn<DvInt4> res;
                                Addition(c1, c2, out a, out b, out res);
                                for (int i = 0; i < res.Length; ++i)
                                    res.Set(i, a[i] + b[i]);
                                return new NumericColumn(res);
                            }
                        case DataKind.R4:
                            {
                                DvInt4[] a;
                                float[] b;
                                DataColumn<float> res;
                                Addition(c1, c2, out a, out b, out res);
                                for (int i = 0; i < res.Length; ++i)
                                    res.Set(i, (int)a[i] + b[i]);
                                return new NumericColumn(res);
                            }
                        default:
                            throw new DataTypeError(string.Format("Addition not implemented for {0}, {1}.", c1.Kind, c2.Kind));
                    }
                case DataKind.R4:
                    switch (c2.Kind)
                    {
                        case DataKind.I4:
                            {
                                float[] a;
                                DvInt4[] b;
                                DataColumn<float> res;
                                Addition(c1, c2, out a, out b, out res);
                                for (int i = 0; i < res.Length; ++i)
                                    res.Set(i, a[i] + (int)b[i]);
                                return new NumericColumn(res);
                            }
                        case DataKind.R4:
                            {
                                float[] a;
                                float[] b;
                                DataColumn<float> res;
                                Addition(c1, c2, out a, out b, out res);
                                for (int i = 0; i < res.Length; ++i)
                                    res.Set(i, a[i] + b[i]);
                                return new NumericColumn(res);
                            }
                        default:
                            throw new DataTypeError(string.Format("Addition not implemented for {0}, {1}.", c1.Kind, c2.Kind));
                    }
                default:
                    throw new DataTypeError(string.Format("Addition not implemented for {0} for left element.", c1.Kind));
            }
        }

        #endregion
    }
}
