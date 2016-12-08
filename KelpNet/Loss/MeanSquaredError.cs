using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet.Loss
{
    public partial class LossFunctions
    {
        public static NdArray MeanSquaredError(NdArray input, NdArray teachSignal, out double loss)
        {
            loss = 0.0;

            double[] diff = new double[teachSignal.Length];
            double coeff = 2.0 / diff.Length;

            for (int i = 0; i < input.Length; i++)
            {
                diff[i] = input.Data[i] - teachSignal.Data[i];
                loss += Math.Pow(diff[i], 2);

                diff[i] *= coeff;
            }

            loss /= diff.Length;

            return new NdArray(diff, teachSignal.Shape);
        }

        public static NdArray[] MeanSquaredError(NdArray[] input, NdArray[] teachSignal, out double loss)
        {
            double[] lossList = new double[input.Length];
            NdArray[] result = new NdArray[input.Length];

            for (int i = 0; i < input.Length; i++)
            {
                result[i] = MeanSquaredError(input[i], teachSignal[i], out lossList[i]);
            }

            loss = lossList.Average();

            return result;
        }
    }
}

