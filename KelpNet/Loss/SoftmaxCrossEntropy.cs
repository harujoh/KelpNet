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

            double[] logY = SoftmaxLog(input.Data);
            loss = -logY[maxIndex];

            double[] gx = new double[logY.Length];

            for (int i = 0; i < logY.Length; i++)
            {
                gx[i] = Math.Exp(logY[i]);
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
            double m = x.Max();

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = Math.Exp(x[i] - m);
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
