using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : NeedPreviousDataFunction
    {
        public Tanh(string name = "Tanh") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int j = 0; j < y.Length; j++)
            {
                y[j] = Math.Tanh(x.Data[j]);
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int j = 0; j < gx.Length; j++)
            {
                gx[j] = gy.Data[j] * (1 - prevOutput.Data[j] * prevOutput.Data[j]);
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
