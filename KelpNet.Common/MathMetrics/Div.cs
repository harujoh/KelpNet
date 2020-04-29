using System;

namespace KelpNet
{
    public class Div<T> : NdArrayAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenSub(T x, T y) { return Operator<T>.Subtract(x, y); }
        static T GenMul(T x, T y) { return Operator<T>.Multiply(x, y); }
        static T GenDiv(T x, T y) { return Operator<T>.Divide(x, y); }

        private const string FUNCTION_NAME = "Div";

        public Div(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, NdArray<T> b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] / b.Data[i];
                resultData[i] = GenDiv(a.Data[i], b.Data[i]);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //T gx = y.Grad[i] / b.Data[i];
                T gx = GenDiv(y.Grad[i], b.Data[i]);

                //a.Grad[i] += gx;
                a.Grad[i] = GenAdd(a.Grad[i], gx);

                //b.Grad[i] -= gx * a.Data[i] / b.Data[i]; //b.Grad[i] += -gx * a.Data[i] / b.Data[i];
                b.Grad[i] = GenSub(b.Grad[i], GenDiv(GenMul(gx, a.Data[i]), b.Data[i]));
            }
        }
    }

    //右辺が定数
    public class DivConst<T> : NdArrayAndConstFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenDiv(T x, T y) { return Operator<T>.Divide(x, y); }

        private const string FUNCTION_NAME = "DivConst";

        public DivConst(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, T b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] / b;
                resultData[i] = GenDiv(a.Data[i], b);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, T b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += y.Grad[i] / b;
                a.Grad[i] = GenAdd(a.Grad[i], GenDiv(y.Grad[i], b));

            }
        }
    }

    //左辺が定数
    public class ConstDiv<T> : ConstAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenSub(T x, T y) { return Operator<T>.Subtract(x, y); }
        static T GenMul(T x, T y) { return Operator<T>.Multiply(x, y); }
        static T GenDiv(T x, T y) { return Operator<T>.Divide(x, y); }

        private const string FUNCTION_NAME = "ConstDiv";

        public ConstDiv(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(T a, NdArray<T> b)
        {
            T[] resultData = new T[b.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a / b.Data[i];
                resultData[i] = GenDiv(a, b.Data[i]);
            }

            return NdArray.Convert(resultData, b.Shape, b.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, T a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //T gx = y.Grad[i] / b.Data[i];
                //b.Grad[i] += -gx * a / b.Data[i];

                //b.Grad[i] -= y.Grad[i] / b.Data[i] * a / b.Data[i];
                b.Grad[i] = GenSub(b.Grad[i], GenDiv(GenMul(GenDiv(y.Grad[i], b.Data[i]), a), b.Data[i]));
            }
        }
    }

}
