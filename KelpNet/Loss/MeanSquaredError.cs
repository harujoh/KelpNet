using System;

namespace KelpNet
{
    public class MeanSquaredError<T> : LossFunction<T> where T : unmanaged, IComparable<T>
    {
        public override Real<T> Evaluate(NdArray<T>[] input, NdArray<T>[] teachSignal)
        {
            Real<T> resultLoss = 0;

#if DEBUG
            if (input.Length != teachSignal.Length) throw new Exception("入力と教師信号のサイズが異なります");
#endif

            for (int k = 0; k < input.Length; k++)
            {
                Real<T> sumLoss = 0;
                RealArray<T> resultArray = new T[input[k].DataLength];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    Real<T> localloss = 0;
                    Real<T> coeff = 2.0f / teachSignal[k].Length;

                    int batchoffset = b * teachSignal[k].Length;

                    for (int i = 0; i < input[k].Length; i++)
                    {
                        Real<T> result = input[k].Data[b * input[k].Length + i] - teachSignal[k].Data[batchoffset + i];
                        localloss += result * result;

                        resultArray[batchoffset + i] = result * coeff;
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