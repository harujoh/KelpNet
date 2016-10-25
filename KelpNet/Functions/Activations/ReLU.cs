using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : NeedPreviousDataFunction
    {
        public ReLU(string name = "ReLU") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int j = 0; j < x.Length; j++)
            {
                y[j] = Math.Max(0, x.Data[j]);
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int j = 0; j < gy.Length; j++)
            {
                gx[j] = prevOutput.Data[j] > 0 ? gy.Data[j] : 0;
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
