using System;

namespace KelpNet.Functions.Activations
{
    public class Sigmoid : NeedPreviousDataFunction
    {
        protected override NdArray ForwardSingle(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = 1 / (1 + Math.Exp(-x.Data[i]));
            }

            return y;
        }

        protected override NdArray BackwardSingle(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = gy.Data[i] * prevOutput.Data[i] * (1 - prevOutput.Data[i]);
            }

            return result;
        }
    }
}
