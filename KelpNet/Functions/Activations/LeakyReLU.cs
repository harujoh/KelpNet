using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousOutputFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU", bool isParallel = true) : base(name,isParallel)
        {
            this._slope = slope;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = x.Data.ToArray();

            for (int i = 0; i < x.Length; i++)
            {
                if (y[i] < 0)
                {
                    y[i] *= this._slope;
                }
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = gy.Data.ToArray();

            for (int i = 0; i < prevOutput.Data.Length; i++)
            {
                if (prevOutput.Data[i] <= 0)
                {
                    gx[i] = prevOutput.Data[i] * this._slope;
                }
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
