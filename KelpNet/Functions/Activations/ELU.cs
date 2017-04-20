using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : NeedPreviousDataFunction
    {
        private readonly double _alpha;

        public ELU(double alpha = 1.0, string name = "ELU", bool isParallel = true) : base(name, isParallel)
        {
            this._alpha = alpha;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = x.Data.ToArray();

            for (int i = 0; i < y.Length; i++)
            {
                if (y[i] < 0)
                {
                    y[i] = this._alpha * (Math.Exp(y[i]) - 1);
                }
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = gy.Data.ToArray();

            for (int i = 0; i < prevOutput.Data.Length; i++)
            {
                if (prevOutput.Data[i] <= 0)
                {
                    gx[i] *= this._alpha * Math.Exp(prevInput.Data[i]);
                }
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
