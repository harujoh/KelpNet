using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    public class Softplus : NeedPreviousOutputFunction
    {
        private readonly Real _beta;
        private readonly Real _betaInv;

        public Softplus(Real? beta = null, string name = "Softplus", bool isGpu = true) : base(name, isGpu)
        {
            this._beta = beta ?? 1;
            this._betaInv = 1 / this._beta;
        }

        public override void InitKernel()
        {
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
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
                    y[i + b * x.Length] = (maxval + (Real)Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * this._beta)))) * this._betaInv;
                }

            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            Real[] gx = new Real[gy.Data.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = (1 - 1 / (1 + (Real)Math.Exp(this._beta * prevOutput.Data[i]))) * gy.Data[i];
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}

