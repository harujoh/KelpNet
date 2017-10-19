using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Softmax : SingleInputFunction
    {
        public Softmax(string name = "Softmax") : base(name)
        {
            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
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

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            Real[] gx = new Real[y.Grad.Length];

            for (int b = 0; b < y.BatchCount; b++)
            {
                Real sumdx = 0;

                for (int i = 0; i < y.Length; i++)
                {
                    gx[i + b * y.Length] = y.Data[i + b * y.Length] * y.Data[i + b * y.Length];
                    sumdx += gx[i + b * y.Length];
                }

                for (int i = 0; i < y.Length; i++)
                {
                    gx[i + b * y.Length] -= y.Data[i + b * y.Length] * sumdx;
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += gx[i];
            }
        }
    }
}
