using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : NeedPreviousOutputFunction
    {
        public Sigmoid(string name = "Sigmoid") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = 1 / (1 + Math.Exp(-x.Data[i]));
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int i = 0; i < gy.Length; i++)
            {
                gx[i] = gy.Data[i] * prevOutput.Data[i] * (1 - prevOutput.Data[i]);
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
