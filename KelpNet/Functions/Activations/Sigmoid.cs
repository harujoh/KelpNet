using System;

namespace KelpNet.Functions.Activations
{
    public class Sigmoid : PredictableFunction
    {
        public override NdArray Forward(NdArray x, int batchID = 0)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = 1 / (1 + Math.Exp(-x.Data[i]));
            }

            return y;
        }

        public override NdArray Backward(NdArray gy, int batchID = 0)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = gy.Data[i] * PrevOutput[batchID].Data[i] * (1 - PrevOutput[batchID].Data[i]);
            }

            return result;
        }
    }
}
