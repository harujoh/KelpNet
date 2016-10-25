using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Softmax : NeedPreviousDataFunction
    {
        public Softmax(string name = "Softmax") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            var maxval = x.Data.Max();
            var sumval = 0.0;

            for (int j = 0; j < y.Length; j++)
            {
                y[j] = Math.Exp(x.Data[j] - maxval);
                sumval += y[j];
            }

            for (int j = 0; j < y.Length; j++)
            {
                y[j] /= sumval;
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];
            var sumdx = 0.0;

            for (int j = 0; j < gx.Length; j++)
            {
                gx[j] = prevOutput.Data[j] * gy.Data[j];
                sumdx += gx[j];
            }

            for (int j = 0; j < gx.Length; j++)
            {
                gx[j] -= prevOutput.Data[j] * sumdx;
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
