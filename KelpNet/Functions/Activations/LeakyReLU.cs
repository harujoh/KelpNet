using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousDataFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU") : base(name)
        {
            this._slope = slope;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int j = 0; j < x.Length; j++)
            {
                if (y[j] < 0) y[j] *= this._slope;
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int j = 0; j < gx.Length; j++)
            {
                gx[j] = prevOutput.Data[j] > 0 ? gy.Data[j] : prevOutput.Data[j] * this._slope;
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
