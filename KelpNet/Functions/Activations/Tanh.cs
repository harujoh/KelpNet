using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : NeedPreviousOutputFunction
    {
        public Tanh(string name = "Tanh") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = Math.Tanh(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = gy.Data[i] * (1 - prevOutput.Data[i] * prevOutput.Data[i]);
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
