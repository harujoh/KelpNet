using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : NeedPreviousOutputFunction
    {
        public Sigmoid(string name = "Sigmoid") : base(name)
        {
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] y = new double[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                y[i] = 1 / (1 + Math.Exp(-x.Data[i]));
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            double[] gx = new double[gy.Data.Length];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                gx[i] = gy.Data[i] * prevOutput.Data[i] * (1 - prevOutput.Data[i]);
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
