using System;

namespace KelpNet
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
                Real[] resultArray = new Real[input[k].Data.Length];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    Real localloss = 0;
                    Real coeff = 2.0 / teachSignal[k].Length;

                    int batchoffset = b * teachSignal[k].Length;

                    for (int i = 0; i < input[k].Length; i++)
                    {
                        Real result = input[k].Data[b * input[k].Length + i] - teachSignal[k].Data[batchoffset + i];
                        localloss += result * result;

                        resultArray[batchoffset + i] = result * coeff / input[k].BatchCount;
                    }

                    sumLoss += localloss / teachSignal[k].Length;
                }

                resultLoss += sumLoss / input[k].BatchCount;

                input[k].Grad = resultArray;
            }

            resultLoss /= input.Length;

            return resultLoss;
        }
    }
}