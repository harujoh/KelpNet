using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Softmax : NeedPreviousOutputFunction
    {
        public Softmax(string name = "Softmax") : base(name)
        {
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = x.Data[b * x.Length];

                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < x.Data[i + b * x.Length])
                    {
                        maxval = x.Data[i + b * x.Length];
                    }
                }

                Real sumval = 0;

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = Math.Exp(x.Data[i + b * x.Length] - maxval);
                    sumval += y[i + b * x.Length];
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] /= sumval;
                }
            }

            return BatchArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput)
        {
            Real[] gx = new Real[gy.Data.Length];

            for (int b = 0; b < gy.BatchCount; b++)
            {
                Real sumdx = 0;

                for (int i = 0; i < gy.Length; i++)
                {
                    gx[i + b * gy.Length] = prevOutput.Data[i + b * prevOutput.Length] * gy.Data[i + b * gy.Length];
                    sumdx += gx[i + b * gy.Length];
                }

                for (int i = 0; i < gy.Length; i++)
                {
                    gx[i + b * gy.Length] -= prevOutput.Data[i + b * prevOutput.Length] * sumdx;
                }
            }

            return BatchArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
