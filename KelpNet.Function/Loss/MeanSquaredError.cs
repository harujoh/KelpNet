using System;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class MeanSquaredError<T> : LossFunction<T, T> where T : unmanaged, IComparable<T>
    {
        public MeanSquaredError()
        {
            switch (this)
            {
                case MeanSquaredError<float> meanSquaredErrorF:
                    meanSquaredErrorF.EvaluateFunc = MeanSquaredErrorF.Evaluate;
                    break;

                case MeanSquaredError<double> meanSquaredErrorD:
                    meanSquaredErrorD.EvaluateFunc = MeanSquaredErrorD.Evaluate;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class MeanSquaredErrorD
#else
    public static class MeanSquaredErrorF
#endif
    {
        public static Real Evaluate(NdArray<Real>[] input, NdArray<Real>[] teachSignal)
        {
            Real resultLoss = 0;

#if DEBUG
            if (input.Length != teachSignal.Length) throw new Exception("入力と教師信号のサイズが異なります");
#endif

            for (int k = 0; k < input.Length; k++)
            {
                Real sumLoss = 0;
                Real[] resultArray = new Real[input[k].Data.Length];
                Real coeff = (Real)2 / teachSignal[k].Length;

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    Real localloss = 0;
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
