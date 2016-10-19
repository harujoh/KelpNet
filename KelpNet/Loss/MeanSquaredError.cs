using System;
using System.Linq;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Loss
{
    public partial class LossFunctions
    {
        public static NdArray MeanSquaredError(NdArray input, NdArray teachSignal, out double loss)
        {
            return MeanSquaredError(new[] { input }, new[] { teachSignal }, out loss)[0];
        }

        public static NdArray[] MeanSquaredError(NdArray[] input, NdArray[] teachSignal, out double loss)
        {
            double[] lossList = new double[input.Length];
            NdArray[] result = new NdArray[input.Length];

#if DEBUG
            for (int i = 0; i < input.Length; i++)
#else
            Parallel.For(0, input.Length, i =>
#endif
            {
                double localloss = 0.0;

                NdArray diff = NdArray.ZerosLike(teachSignal[i]);
                double coeff = 2.0 / diff.Length;

                for (int j = 0; j < input[i].Length; j++)
                {
                    diff.Data[j] = input[i].Data[j] - teachSignal[i].Data[j];
                    localloss += Math.Pow(diff.Data[j], 2);

                    diff.Data[j] *= coeff;
                }

                localloss /= diff.Length;

                lossList[i] = localloss;

                result[i] = diff;
            }

#if !DEBUG
            );
#endif
            loss = lossList.Average();

            return result;
        }
    }
}

