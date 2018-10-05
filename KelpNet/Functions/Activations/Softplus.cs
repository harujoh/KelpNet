using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    public class Softplus : SingleInputFunction
    {
        const string FUNCTION_NAME = "Softplus";

        private readonly Real _beta;
        private readonly Real _betaInv;

        public Softplus(Real beta = 1.0f, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._beta = beta;
            this._betaInv = 1.0f / this._beta;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = x.Data[i + b * x.Length] * this._beta;
                }

                Real maxval = y[b * x.Length];
                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < y[i + b * x.Length])
                    {
                        maxval = y[i + b * x.Length];
                    }
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = (Real)(maxval + Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * this._beta)))) * this._betaInv;
                }

            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += (Real)(1.0 - 1.0 / (1.0 + Math.Exp(this._beta * y.Data[i]))) * y.Grad[i];
            }

        }
    }
}

