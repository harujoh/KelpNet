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

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = Math.Exp(x.Data[i] - maxval);
                sumval += y[i];
            }

            for (int i = 0; i < x.Length; i++)
            {
                y[i] /= sumval;
            }

            return NdArray.FromArray(y);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];
            var sumdx = 0.0;

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = prevOutput.Data[i] * gy.Data[i];
                sumdx += gx[i];
            }


            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] -= prevOutput.Data[i] * sumdx;
            }

            return NdArray.FromArray(gx);
        }
    }
}
