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
            loss = -logY.Get(0, maxIndex);

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

            double[] m = new double[x.Shape[0]];
            double[] xbuf = new double[x.Shape[1]];
            NdArray y = NdArray.EmptyLike(x);
            double[] ybuf = new double[x.Shape[1]];

            for (int i = 0; i < m.Length; i++)
            {
                Buffer.BlockCopy(x.Data, xbuf.Length * i, xbuf, 0, sizeof(double) * xbuf.Length);
                m[i] = xbuf.Max();

                for (int j = 0; j < x.Shape[1]; j++)
                {
                    y.Data[y.GetIndex(i, j)] = Math.Exp(x.Get(i, j) - m[i]);
                }

                Buffer.BlockCopy(y.Data, ybuf.Length * i, ybuf, 0, sizeof(double) * ybuf.Length);
                m[i] += Math.Log(ybuf.Sum());
            }

            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    result.Data[result.GetIndex(i, j)] = x.Get(i, j) - m[i];
                }
            }

            return result;
        }
    }
}
