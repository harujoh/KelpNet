using System;

namespace KelpNet.Functions.Activations
{
    public class ReLU : Function, IPredictableFunction
    {
        public override NdArray Forward(NdArray x)
        {
            PrevOutput = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                PrevOutput.Data[i] = Math.Max(0, x.Data[i]);
            }

            return PrevOutput;
        }

        public override NdArray Backward(NdArray gy)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = PrevOutput.Data[i] > 0 ? gy.Data[i] : 0;
            }

            return result;
        }

        public NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}
