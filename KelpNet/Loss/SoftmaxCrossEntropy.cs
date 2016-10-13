using System;
using System.Linq;
using KelpNet.Common;

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

            for (int i = 0; i < logY.Length; i++)
            {
                result.Data[i] = Math.Exp(logY.Data[i]);
            }

            result.Data[maxIndex] -= 1;

            return result;
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
