using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    public class SoftmaxCrossEntropy : ILossFunction
    {
        public BatchArray Evaluate(BatchArray input, BatchArray teachSignal, out double loss)
        {
            double[] localloss = new double[input.BatchCount];
            double[] gx = new double[input.Data.Length];

            for (int b = 0; b < input.BatchCount; b++)
            {
                double maxIndex = 0;

                for (int i = 0; i < teachSignal.Length; i++)
                {
                    if (maxIndex < teachSignal.Data[i + b * teachSignal.Length])
                    {
                        maxIndex = teachSignal.Data[i + b * teachSignal.Length];
                    }
                }

                double[] logY = new double[input.Length];
                double[] y = new double[input.Length];
                double m = input.Data[b * input.Length];

                for (int i = 1; i < input.Length; i++)
                {
                    if (m < input.Data[i + b * input.Length])
                    {
                        m = input.Data[i + b * input.Length];
                    }
                }

                for (int i = 0; i < input.Length; i++)
                {
                    y[i] = Math.Exp(input.Data[i + b * input.Length] - m);
                }

                m += Math.Log(y.Sum());

                for (int i = 0; i < input.Length; i++)
                {
                    logY[i] = input.Data[i + b * input.Length] - m;
                }

                localloss[b] = -logY[(int)maxIndex];


                for (int i = 0; i < logY.Length; i++)
                {
                    gx[i + b * input.Length] = Math.Exp(logY[i]);
                }

                gx[(int)maxIndex + b * input.Length] -= 1;
            }

            loss = localloss.Average();

            return BatchArray.Convert(gx, input.Shape, input.BatchCount);
        }
    }
}
