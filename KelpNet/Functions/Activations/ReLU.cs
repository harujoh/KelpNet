using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

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
            double[] y = x.Data.ToArray();

            for (int i = 0; i < y.Length; i++)
            {
                if (y[i] < 0)
                {
                    y[i] = 0;
                }
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput)
        {
            double[] gx = gy.Data.ToArray();

            for (int i = 0; i < prevOutput.Data.Length; i++)
            {
                if (prevOutput.Data[i] <= 0)
                {
                    gx[i] = 0.0;
                }
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
