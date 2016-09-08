using System;

namespace KelpNet.Functions.Activations
{
    public class Tanh : PredictableFunction
    {
        protected override NdArray ForwardSingle(NdArray x, int batchID = 0)
        {
            NdArray y = NdArray.EmptyLike(x);

            for (int i = 0; i < x.Length; i++)
            {
                y.Data[i] = Math.Tanh(x.Data[i]);
            }

            return y;
        }

        public override NdArray Backward(NdArray gy, int batchID = 0)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                result.Data[i] = gy.Data[i] * (1 - PrevOutput[batchID].Data[i] * PrevOutput[batchID].Data[i]);
            }

            return result;
        }
    }
}
