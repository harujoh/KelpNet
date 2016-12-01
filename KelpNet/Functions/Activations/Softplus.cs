using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    public class Softplus : NeedPreviousOutputFunction
    {
        private readonly double _beta;
        private readonly double _betaInv;

        public Softplus(double beta = 1.0, string name = "Softplus") : base(name)
        {
            this._beta = beta;
            this._betaInv = 1.0 / beta;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = x.Data[i] * this._beta;
            }

            double maxval = Math.Max(y.Max(), 0);

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = (maxval + Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i] * this._beta)))) * this._betaInv;
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = (1 - 1 / (1 + Math.Exp(this._beta * prevOutput.Data[i]))) * gy.Data[i];
            }

            return new NdArray(gx, gy.Shape);
        }

    }
}
