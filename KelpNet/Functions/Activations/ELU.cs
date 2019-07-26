using System;

namespace KelpNet
{
    [Serializable]
    public class ELU<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "ELU";

        private readonly Real<T> _alpha;

        public ELU(double alpha = 1.0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._alpha = alpha;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        private NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            RealArray<T> result = new T[x.DataLength];

            for (int i = 0; i < x.DataLength; i++)
            {
                if (x.Data[i] >= 0)
                {
                    result[i] = x.Data[i];
                }
                else
                {
                    result[i] = this._alpha * (Math.Exp(x.Data[i]) - 1.0);
                }
            }

            return NdArray<T>.Convert(result, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                if (x.Data[i] >= 0)
                {
                    x.Grad[i] += y.Grad[i];
                }
                else
                {
                    x.Grad[i] += y.Grad[i] * this._alpha * Math.Exp(x.Data[i]);
                }
            }
        }
    }
}
