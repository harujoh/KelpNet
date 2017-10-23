using System;
using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    public class MeanSquaredError : LossFunction
    {
        public override Real Evaluate(NdArray[] input, NdArray[] teachSignal)
        {
            Real resultLoss = 0;

#if DEBUG
            if (input.Length != teachSignal.Length) throw new Exception("入力と教師信号のサイズが異なります");
#endif

            for (int k = 0; k < input.Length; k++)
            {
                Real sumLoss = 0;
                Real[] result = new Real[input[k].Data.Length];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    Real localloss = 0;
                    Real coeff = 2.0 / teachSignal[k].Length;

                    for (int i = 0; i < input[k].Length; i++)
                    {
                        result[i + b * teachSignal[k].Length] = input[k].Data[i + b * input[k].Length] - teachSignal[k].Data[i + b * teachSignal[k].Length];
                        localloss += result[i + b * teachSignal[k].Length] * result[i + b * teachSignal[k].Length];

                        result[i + b * teachSignal[k].Length] *= coeff;
                    }

                    sumLoss += localloss / teachSignal[k].Length;
                }

                resultLoss += sumLoss / input[k].BatchCount;

                input[k].Grad = result;
            }

            resultLoss /= input.Length;

            return resultLoss;
        }
    }
}