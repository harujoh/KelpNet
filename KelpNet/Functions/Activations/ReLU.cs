using System;

namespace KelpNet.Functions.Activations
{
    public class ReLU : Function, IPredictableFunction
    {
        public override NdArray Forward(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = Math.Max(0, x.Data[i]);
            }

            return y;
        }

        public override NdArray Backward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = prevOutput.Data[i] > 0 ? gy.Data[i] : 0;
            }

            return result;
        }

        public NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}
