using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : NeedPreviousDataFunction
    {
        public Sigmoid(string name = "Sigmoid") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int j = 0; j < x.Length; j++)
            {
                y[j] = 1 / (1 + Math.Exp(-x.Data[j]));
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int j = 0; j < gy.Length; j++)
            {
                gx[j] = gy.Data[j] * prevOutput.Data[j] * (1 - prevOutput.Data[j]);
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
