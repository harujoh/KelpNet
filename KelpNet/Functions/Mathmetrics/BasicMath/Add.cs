using System;

namespace KelpNet
{
    public class Add<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Add";

        public Add(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            RealArray<T> resultData = new T[a.DataLength];

            for (int i = 0; i < a.DataLength; i++)
            {
                resultData[i] = a.Data[i] + b.Data[i];
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
                b.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }

    public class AddConst<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "AddConst";

        public AddConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            RealArray<T> resultData = new T[a.DataLength];
            Real<T> val = b.Data[0];

            for (int i = 0; i < a.DataLength; i++)
            {
                resultData[i] = a.Data[i] + val;
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }

}
