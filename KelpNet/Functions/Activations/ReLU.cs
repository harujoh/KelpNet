using System;

namespace KelpNet.Functions.Activations
{
    public class ReLU : PredictableFunction
    {
        protected override NdArray ForwardSingle(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = Math.Max(0, x.Data[i]);
            }

            return y;
        }

        protected override NdArray BackwardSingle(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : 0;
            }

            return result;
        }
    }
}
