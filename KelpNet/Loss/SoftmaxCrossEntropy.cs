using System;
using System.Linq;

namespace KelpNet.Loss
{
    public partial class LossFunctions
    {
        public static NdArray SoftmaxCrossEntropy(NdArray input, NdArray teachSignal, out double loss)
        {
            int maxIndex = (int)Math.Max(teachSignal.Data.Max(), 0.0);

            var logY = SoftmaxLog(input);
            loss = -logY.Get(maxIndex);

            NdArray result = NdArray.EmptyLike(logY);

            for (int i = 0; i < logY.Length; i++)
            {
                result.Data[i] = Math.Exp(logY.Data[i]);
            }

            result.Data[maxIndex] -= 1;

            return result;
        }


        static NdArray SoftmaxLog(NdArray x)
        {
            NdArray result = NdArray.EmptyLike(x);

            //double[] xbuf = new double[x.Length];
            NdArray y = NdArray.EmptyLike(x);
            //double[] ybuf = new double[x.Length];

            //Buffer.BlockCopy(x.Data, xbuf.Length, xbuf, 0, sizeof(double) * xbuf.Length);
            //var m = xbuf.Max();
            var m = x.Data.Max();

            for (int j = 0; j < x.Length; j++)
            {
                y.Data[j] = Math.Exp(x.Data[j] - m);
            }

            //Buffer.BlockCopy(y.Data, ybuf.Length, ybuf, 0, sizeof(double) * ybuf.Length);
            //m += Math.Log(ybuf.Sum());
            m += Math.Log(y.Data.Sum());

            for (int i = 0; i < x.Length; i++)
            {
                result.Data[i] = x.Data[i] - m;
            }

            return result;
        }
    }
}
