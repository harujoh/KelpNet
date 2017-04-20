using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    public class SoftmaxCrossEntropy : ILossFunction
    {
        public NdArray Evaluate(NdArray input, NdArray teachSignal, out double loss)
        {
            int maxIndex = (int)Math.Max(teachSignal.Data.Max(), 0.0);

            double[] logY = this.SoftmaxLog(input.Data);
            loss = -logY[maxIndex];

            double[] gx = new double[logY.Length];

            for (int i = 0; i < logY.Length; i++)
            {
                gx[i] = Math.Exp(logY[i]);
            }

            gx[maxIndex] -= 1;

            return NdArray.Convert(gx, input.Shape);
        }

        public NdArray[] Evaluate(NdArray[] input, NdArray[] teachSignal, out double loss)
        {
            double[] localloss = new double[input.Length];
            NdArray[] resultArray = new NdArray[input.Length];
            
            for(int i = 0; i < input.Length; i ++)
            {
                resultArray[i] = this.Evaluate(input[i], teachSignal[i], out localloss[i]);
            }

            loss = localloss.Average();
            return resultArray;
        }

        private double[] SoftmaxLog(double[] x)
        {
            double[] result = new double[x.Length];

            double y = 0;
            double m = x.Max();

            for (int i = 0; i < x.Length; i++)
            {
                y += Math.Exp(x[i] - m);
            }

            m += Math.Log(y);

            for (int i = 0; i < x.Length; i++)
            {
                result[i] = x[i] - m;
            }

            return result;
        }
    }
}
