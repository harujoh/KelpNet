using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public abstract class CompressibleActivation<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Activation";

        //.Netで使用するActivateの仮想関数
        internal abstract Real<T> ForwardActivate(Real<T> x);
        internal abstract Real<T> BackwardActivate(Real<T> gy, Real<T> y);

        protected string ActivateKernelString;

        protected CompressibleActivation(string functionName, KeyValuePair<string, string>[] parameters, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.SingleInputForward = this.NeedPreviousForwardCpu;
            this.SingleOutputBackward = this.NeedPreviousBackwardCpu;
        }

        private NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            Real<T>[] y = new Real<T>[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = this.ForwardActivate(x.Data[i]);
            }

            return NdArray<T>.Convert(y, x.Shape, x.BatchCount, this);
        }

        private void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += this.BackwardActivate(y.Grad[i], y.Data[i]);
            }
        }
    }
}
