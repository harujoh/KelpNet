using System;

namespace KelpNet
{
    public class Softplus<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Softplus";

        private readonly Real<T> _beta;
        private readonly Real<T> _betaInv;

        public Softplus(double beta = 1.0, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._beta = beta;
            this._betaInv = 1.0f / this._beta;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            Real<T>[] y = new Real<T>[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = x.Data[i + b * x.Length] * this._beta;
                }

                Real<T> maxval = y[b * x.Length];
                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < y[i + b * x.Length])
                    {
                        maxval = y[i + b * x.Length];
                    }
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = (maxval + Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * this._beta)))) * this._betaInv;
                }

            }

            return NdArray<T>.Convert(y, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += (1.0 - 1.0 / (1.0 + Math.Exp(this._beta * y.Data[i]))) * y.Grad[i];
            }

        }
    }
}

