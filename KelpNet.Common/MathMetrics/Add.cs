using System;

namespace KelpNet
{
    public class Add<T> : NdArrayAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        //ジェネリックを無理やり計算できるようにする
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }

        private const string FUNCTION_NAME = "Add";

        public Add(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, NdArray<T> b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] + b.Data[i];
                resultData[i] = GenAdd(a.Data[i], b.Data[i]);
            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += y.Grad[i]; // * 1.0f
                a.Grad[i] = GenAdd(a.Grad[i], y.Grad[i]);
                //b.Grad[i] += y.Grad[i]; // * 1.0f
                b.Grad[i] = GenAdd(b.Grad[i], y.Grad[i]);
            }
        }
    }

    public class AddConst<T> : NdArrayAndConstFunction<T> where T : unmanaged, IComparable<T>
    {
        //ジェネリックを無理やり計算できるようにする
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }

        private const string FUNCTION_NAME = "AddConst";

        public AddConst(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(NdArray<T> a, T b)
        {
            T[] resultData = new T[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a.Data[i] + b;
                resultData[i] = GenAdd(a.Data[i], b);

            }

            return NdArray.Convert(resultData, a.Shape, a.BatchCount, this);
        }

        public override void DualOutputBackward(NdArray<T> y, NdArray<T> a, T b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //a.Grad[i] += y.Grad[i]; // * 1.0f
                a.Grad[i] = GenAdd(a.Grad[i], y.Grad[i]);
            }
        }
    }

    public class ConstAdd<T> : ConstAndNdArrayFunction<T> where T : unmanaged, IComparable<T>
    {
        //ジェネリックを無理やり計算できるようにする
        static T GenAdd(T x, T y) { return Operator<T>.Add(x, y); }

        private const string FUNCTION_NAME = "AddConst";

        public ConstAdd(string name = FUNCTION_NAME) : base(name)
        {
        }

        public override NdArray<T> DualInputForward(T a, NdArray<T> b)
        {
            T[] resultData = new T[b.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                //resultData[i] = a + b.Data[i];
                resultData[i] = GenAdd(a, b.Data[i]);
            }

            return NdArray.Convert(resultData, b.Shape, b.BatchCount, this);
        }


        public override void DualOutputBackward(NdArray<T> y, T a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                //b.Grad[i] += y.Grad[i]; // * 1.0f
                b.Grad[i] = GenAdd(b.Grad[i], y.Grad[i]);
            }
        }
    }

}