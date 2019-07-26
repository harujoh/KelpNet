using System;

namespace KelpNet
{
    public class Mul<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Mul";

        public Mul(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            RealArray<T> resultData = new T[a.DataLength];

            for (int i = 0; i < a.DataLength; i++)
            {
                resultData[i] = a.Data[i] * b.Data[i];
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                a.Grad[i] += b.Data[i] * y.Grad[i];
                b.Grad[i] += a.Data[i] * y.Grad[i];
            }
        }
    }

    public class MulConst<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "MulConst";

        public MulConst(string name = FUNCTION_NAME) : base(name)
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
                resultData[i] = a.Data[i] * val;
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                a.Grad[i] += b.Data[0] * y.Grad[i];
            }
        }
    }

}
