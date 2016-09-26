using System;

namespace KelpNet.Functions.Activations
{
    public class Tanh : NeedPreviousDataFunction
    {
        public Tanh(string name = "Tanh") : base(name)
        {
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = Math.Tanh(x.Data[i]);
            }

            return y;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = gy.Data[i] * (1 - prevOutput.Data[i] * prevOutput.Data[i]);
            }

            return result;
        }
    }
}
