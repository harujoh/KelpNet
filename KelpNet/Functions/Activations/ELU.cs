using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : NeedPreviousDataFunction
    {
        private readonly double _alpha;

        public ELU(double alpha = 1.0 , string name = "ELU") : base(name)
        {
            this._alpha = alpha;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = x.Data[i] >= 0 ? x.Data[i] : this._alpha * (Math.Exp(x.Data[i]) - 1);
            }

            return y;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = prevOutput.Data[i] >= 0 ? gy.Data[i] : gy.Data[i] * this._alpha * Math.Exp(prevInput.Data[i]);
            }

            return result;
        }
    }
}
