using System;

namespace KelpNet
{
    [Serializable]
    public class Swish<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Swish";

        public Swish(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        protected NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            Real<T>[] result = new Real<T>[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] / (1.0 + Math.Exp(-x.Data[i]));
            }

            return NdArray<T>.Convert(result, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i] * (y.Data[i] + y.Data[i] / x.Data[i] * (1 - y.Data[i]));
            }
        }
    }
}
