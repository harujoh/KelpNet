using System;

namespace KelpNet
{
    public class Sub<T> : NdArrayAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenSub(T x, T y) { return Operator<T>.Subtract(x, y); }

        private const string FUNCTION_NAME = "Sub";

        public Sub(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, NdArray<T> b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] - b.Data[i];
                resultData[i] = GenSub(a.Data[i], b.Data[i]);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += y.Grad[i];
                a.Grad[i] = GenAdd(a.Grad[i], y.Grad[i]);
                //b.Grad[i] -= y.Grad[i];
                b.Grad[i] = GenSub(b.Grad[i], y.Grad[i]);
            }
        }
    }

    //右辺が定数
    public class SubConst<T> : NdArrayAndConstFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }
        static T GenSub(T x, T y) { return Operator<T>.Subtract(x, y); }

        private const string FUNCTION_NAME = "SubConst";

        public SubConst(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, T b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] - b;
                resultData[i] = GenSub(a.Data[i], b);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, T b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += y.Grad[i];
                a.Grad[i] = GenAdd(a.Grad[i], y.Grad[i]);
            }
        }
    }

    //左辺が定数
    public class ConstSub<T> : ConstAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        static T GenSub(T x, T y) { return Operator<T>.Subtract(x, y); }

        private const string FUNCTION_NAME = "ConstSub";

        public ConstSub(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(T a, NdArray<T> b)
        {
            T[] resultData = new T[b.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a - b.Data[i];
                resultData[i] = GenSub(a, b.Data[i]);
            }

            return NdArray.Convert(resultData, b.Shape, b.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, T a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //b.Grad[i] -= y.Grad[i];
                b.Grad[i] = GenSub(b.Grad[i], y.Grad[i]);
            }
        }
    }

}
