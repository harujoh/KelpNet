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
            NeedPreviousForward = NeedPreviousForwardCpu;
            NeedPreviousBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray NeedPreviousForwardCpu(NdArray x)
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

            return NdArray.Convert(y, x.Shape, x.BatchCount);
        }

        protected NdArray NeedPreviousBackwardCpu(NdArray gy, NdArray prevOutput)
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

            return NdArray.Convert(gx, gy.Shape, gy.BatchCount);
        }
    }
}
