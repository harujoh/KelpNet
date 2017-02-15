using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousOutputFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU") : base(name)
        {
            this._slope = slope;
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] y = new double[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                y[i] = x.Data[i] < 0 ? x.Data[i] *= this._slope : x.Data[i];
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            double[] gx = new double[gy.Data.Length];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                gx[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : prevOutput.Data[i] * this._slope;
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
