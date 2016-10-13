using System;
using KelpNet.Common;

namespace KelpNet.Loss
{
    public partial class LossFunctions
    {
        public static NdArray MeanSquaredError(NdArray input, NdArray teachSignal, out double loss)
        {
            loss = 0;

            NdArray diff = NdArray.ZerosLike(teachSignal);
            double coeff = 2.0 / diff.Length;

            for (int i = 0; i < input.Length; i++)
            {
                diff.Data[i] = input.Data[i] - teachSignal.Data[i];
                loss += Math.Pow(diff.Data[i], 2);

                diff.Data[i] *= coeff;
            }

            loss /= diff.Length;

            return diff;
        }
    }
}
