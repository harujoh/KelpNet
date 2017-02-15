using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    public class MeanSquaredError : ILossFunction
    {
        public BatchArray Evaluate(BatchArray input, BatchArray teachSignal, out double loss)
        {
            double[] lossList = new double[input.BatchCount];
            double[] result = new double[teachSignal.Data.Length];

            for (int b = 0; b < input.BatchCount; b++)
            {
                loss = 0.0;

                double coeff = 2.0 / teachSignal.Length;

                for (int i = 0; i < input.Length; i++)
                {
                    result[i + b * teachSignal.Length] = input.Data[i + b * input.Length] - teachSignal.Data[i + b * teachSignal.Length];
                    loss += Math.Pow(result[i + b * teachSignal.Length], 2);

                    result[i + b * teachSignal.Length] *= coeff;
                }

                lossList[b] = loss / teachSignal.Length;
            }

            loss = lossList.Average();

            return BatchArray.Convert(result, teachSignal.Shape, teachSignal.BatchCount);
        }
    }
}

