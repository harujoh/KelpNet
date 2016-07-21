using System;

namespace KelpNet.Loss
{
    public partial class LossFunctions
    {
        static public NdArray MeanSquaredError(NdArray input, NdArray teachSignal, out double loss)
        {
            loss = 0;

            NdArray diff = NdArray.EmptyLike(teachSignal);

            for (int i = 0; i < input.Length; i++)
            {
                diff.Data[i] = input.Data[i] - teachSignal.Data[i];
                loss += Math.Pow(diff.Data[i], 2);

                diff.Data[i] *= 2.0 / diff.Length;
            }

            loss /= diff.Length;

            return diff;
        }
    }
}
