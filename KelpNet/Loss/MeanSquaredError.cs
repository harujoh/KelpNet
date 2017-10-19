using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    public class MeanSquaredError : ILossFunction
    {
        public NdArray Evaluate(NdArray input, NdArray teachSignal, out Real loss)
        {
            Real sumLoss = 0;
            Real[] result = new Real[input.Data.Length];

            for (int b = 0; b < input.BatchCount; b++)
            {
                loss = 0;

                Real coeff = 2.0 / teachSignal.Length;

                for (int i = 0; i < input.Length; i++)
                {
                    result[i + b * teachSignal.Length] = input.Data[i + b * input.Length] - teachSignal.Data[i + b * teachSignal.Length];
                    loss += result[i + b * teachSignal.Length] * result[i + b * teachSignal.Length];

                    result[i + b * teachSignal.Length] *= coeff;
                }

                sumLoss += loss / teachSignal.Length;
            }

            loss = sumLoss / input.BatchCount;

            input.Grad = result;
            return input;
        }
    }
}