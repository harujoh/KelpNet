using System;
using KelpNet.Common;

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
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = x.Data[i] >= 0 ? x.Data[i] : this._alpha * (Math.Exp(x.Data[i]) - 1);
            }

            return NdArray.Convert(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = prevOutput.Data[i] >= 0
                    ? gy.Data[i]
                    : gy.Data[i] * this._alpha * Math.Exp(prevInput.Data[i]);
            }

            return NdArray.Convert(gx, gy.Shape);
        }
    }
}
