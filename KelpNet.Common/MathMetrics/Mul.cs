using System;

namespace KelpNet
{
    public class Mul<T> : NdArrayAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenMul(T x, T y) { return Operator<T>.Multiply(x, y); }

        private const string FUNCTION_NAME = "Mul";

        public Mul(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, NdArray<T> b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] * b.Data[i];
                resultData[i] = GenMul(a.Data[i], b.Data[i]);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += b.Data[i] * y.Grad[i];
                a.Grad[i] = GenAdd(a.Grad[i], GenMul(b.Data[i], y.Grad[i]));
                //b.Grad[i] += a.Data[i] * y.Grad[i];
                b.Grad[i] = GenAdd(b.Grad[i], GenMul(a.Data[i], y.Grad[i]));
            }
        }
    }

    public class MulConst<T> : NdArrayAndConstFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenMul(T x, T y) { return Operator<T>.Multiply(x, y); }

        private const string FUNCTION_NAME = "MulConst";

        public MulConst(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, T b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] * b;
                resultData[i] = GenMul(a.Data[i], b);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, T b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += b * y.Grad[i];
                a.Grad[i] = GenAdd(a.Grad[i], GenMul(b, y.Grad[i]));
            }
        }
    }

    public class ConstMul<T> : ConstAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenMul(T x, T y) { return Operator<T>.Multiply(x, y); }

        private const string FUNCTION_NAME = "Mul";

        public ConstMul(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T> DualInputForward(T a, NdArray<T> b)
        {
            T[] resultData = new T[b.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a * b.Data[i];
                resultData[i] = GenMul(a, b.Data[i]);

            }

            return NdArray.Convert(resultData, b.Shape, b.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, T a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //b.Grad[i] += a * y.Grad[i];
                b.Grad[i] = GenAdd(b.Grad[i], GenMul(a, y.Grad[i]));
            }
        }
    }

}
