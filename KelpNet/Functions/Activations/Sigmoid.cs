using System;
using KelpNet.Common;

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : NeedPreviousDataFunction
    {
        public Sigmoid(string name = "Sigmoid") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            NdArray y = NdArray.ZerosLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = 1 / (1 + Math.Exp(-x.Data[i]));
            }

            return y;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.ZerosLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = gy.Data[i] * prevOutput.Data[i] * (1 - prevOutput.Data[i]);
            }

            return result;
        }
    }
}
