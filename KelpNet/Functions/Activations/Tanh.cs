using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Tanh : NeedPreviousOutputFunction
    {
        public Tanh(string name = "Tanh") : base(name)
        {
        }

        protected override BatchArray NeedPreviousForward(BatchArray x, bool isGpu)
        {
            double[] y = new double[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = Math.Tanh(x.Data[i]);
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput, bool isGpu)
        {
            double[] gx = new double[gy.Data.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = gy.Data[i] * (1 - prevOutput.Data[i] * prevOutput.Data[i]);
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
