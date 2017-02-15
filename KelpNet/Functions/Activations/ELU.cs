using System;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : NeedPreviousDataFunction
    {
        private readonly double _alpha;

        public ELU(double alpha = 1.0, string name = "ELU") : base(name)
        {
            this._alpha = alpha;
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] result = new double[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] >= 0 ? x.Data[i] : this._alpha * (Math.Exp(x.Data[i]) - 1);
            }

            return BatchArray.Convert(result, x.Shape, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput, BatchArray prevOutput)
        {
            double[] result = new double[gy.Data.Length];

            for (int i = 0; i < gy.Data.Length; i++)
            {
                result[i] = prevOutput.Data[i] >= 0
                    ? gy.Data[i]
                    : gy.Data[i] * this._alpha * Math.Exp(prevInput.Data[i]);
            }

            return BatchArray.Convert(result, gy.Shape, gy.BatchCount);
        }
    }
}
