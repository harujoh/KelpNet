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

            var logY = SoftmaxLog(input.Data);
            loss = -logY[maxIndex];

            double[] gx = new double[logY.Length];

            for (int j = 0; j < logY.Length; j++)
            {
                gx[j] = Math.Exp(logY[j]);
            }

            gx[maxIndex] -= 1;

            return new NdArray(gx, input.Shape);
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

        static double[] SoftmaxLog(double[] x)
        {
            double[] result = new double[x.Length];

            double[] y = new double[x.Length];
            var m = x.Max();

            for (int j = 0; j < x.Length; j++)
            {
                y[j] = Math.Exp(x[j] - m);
            }

            m += Math.Log(y.Sum());

            for (int i = 0; i < x.Length; i++)
            {
                result[i] = x[i] - m;
            }

            return result;
        }
    }
}
