using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : NeedPreviousDataFunction
    {
        private readonly double _alpha;

        public ELU(double alpha = 1.0, string name = "ELU", int batchCount = 1) : base(name)
        {
            this._alpha = alpha;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] y = new double[x.Length];

            for (int j = 0; j < x.Length; j++)
            {
                y[j] = x.Data[j] >= 0 ? x.Data[j] : this._alpha * (Math.Exp(x.Data[j]) - 1);
            }

            return new NdArray(y, x.Shape);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            double[] gx = new double[gy.Length];

            for (int j = 0; j < gx.Length; j++)
            {
                gx[j] = prevOutput.Data[j] >= 0
                    ? gy.Data[j]
                    : gy.Data[j] * this._alpha * Math.Exp(prevInput.Data[j]);
            }

            return new NdArray(gx, gy.Shape);
        }
    }
}
