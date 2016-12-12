using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet.Loss
{
    public class MeanSquaredError : ILossFunction
    {
        public NdArray Evaluate(NdArray input, NdArray teachSignal, out double loss)
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

            return NdArray.Convert(diff, teachSignal.Shape);
        }

        public NdArray[] Evaluate(NdArray[] input, NdArray[] teachSignal, out double loss)
        {
            double[] lossList = new double[input.Length];
            NdArray[] result = new NdArray[input.Length];

            for (int i = 0; i < input.Length; i++)
            {
                result[i] = this.Evaluate(input[i], teachSignal[i], out lossList[i]);
            }

            loss = lossList.Average();

            return result;
        }
    }
}

