using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    public class Softplus : NeedPreviousOutputFunction
    {
        const string FUNCTION_NAME = "Softplus";

        private readonly Real _beta;
        private readonly Real _betaInv;

        public Softplus(double beta = 1, string name = FUNCTION_NAME) : base(name)
        {
            this._beta = beta;
            this._betaInv = 1 / this._beta;

            NeedPreviousForward = NeedPreviousForwardCpu;
            NeedPreviousBackward = NeedPreviousBackwardCpu;
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
                    y[i + b * x.Length] = (maxval + Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * this._beta)))) * this._betaInv;
                }

            }

            return NdArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected NdArray NeedPreviousBackwardCpu(NdArray gy, NdArray prevOutput)
        {
            Real[] gx = new Real[gy.Data.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = (1 - 1 / (1 + Math.Exp(this._beta * prevOutput.Data[i]))) * gy.Data[i];
            }

            return NdArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}

