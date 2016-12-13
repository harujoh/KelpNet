using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ReLU : NeedPreviousOutputFunction
    {
        public ReLU(string name = "ReLU", bool isParallel = true) : base(name, isParallel)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = Math.Max(0, x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int i = 0; i < gy.Length; i++)
            {
                gx[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : 0;
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
