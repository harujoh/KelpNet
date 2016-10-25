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
        public static NdArray SoftmaxCrossEntropy(NdArray input, NdArray teachSignal, out double loss)
        {
            int maxIndex = (int)Math.Max(teachSignal.Data.Max(), 0.0);

            var logY = SoftmaxLog(input);
            loss = -logY.Data[maxIndex];

            NdArray result = NdArray.ZerosLike(logY);

            for (int j = 0; j < logY.Length; j++)
            {
                result.Data[j] = Math.Exp(logY.Data[j]);
            }

            result.Data[maxIndex] -= 1;

            return result;
        }

        public static NdArray[] SoftmaxCrossEntropy(NdArray[] input, NdArray[] teachSignal, out double loss)
        {
            double[] localloss = new double[input.Length];
            NdArray[] resultArray = new NdArray[input.Length];

#if DEBUG
            for(int i = 0; i < input.Length; i ++)
#else
            Parallel.For(0, input.Length, i =>
#endif
            {
                resultArray[i] = SoftmaxCrossEntropy(input[i], teachSignal[i], out localloss[i]);
            }
#if !DEBUG
            );
#endif
            loss = localloss.Average();
            return resultArray;
        }


        static NdArray SoftmaxLog(NdArray x)
        {
            NdArray result = NdArray.ZerosLike(x);

            NdArray y = NdArray.ZerosLike(x);
            var m = x.Data.Max();

            for (int j = 0; j < x.Length; j++)
            {
                y.Data[j] = Math.Exp(x.Data[j] - m);
            }

            m += Math.Log(y.Data.Sum());

            for (int i = 0; i < x.Length; i++)
            {
                result.Data[i] = x.Data[i] - m;
            }

            return result;
        }
    }
}
