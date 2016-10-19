using System;
using System.Linq;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Softmax : NeedPreviousDataFunction
    {
        public Softmax(string name = "Softmax") : base(name)
        {
        }

        protected override NdArray[] NeedPreviousForward(NdArray[] x)
        {
            NdArray[] result = new NdArray[x.Length];

#if DEBUG
            for (int i = 0; i < x.Length; i++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                double[] y = new double[x[i].Length];

                var maxval = x[i].Data.Max();
                var sumval = 0.0;

                for (int j = 0; j < y.Length; j++)
                {
                    y[j] = Math.Exp(x[i].Data[j] - maxval);
                    sumval += y[j];
                }

                for (int j = 0; j < y.Length; j++)
                {
                    y[j] /= sumval;
                }

                result[i] = new NdArray(y, x[i].Shape);
            }
#if !DEBUG
            );
#endif

            return result;
        }

        protected override NdArray[] NeedPreviousBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput)
        {
            NdArray[] result = new NdArray[gy.Length];

#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                double[] gx = new double[gy[i].Length];
                var sumdx = 0.0;

                for (int j = 0; j < gx.Length; j++)
                {
                    gx[j] = prevOutput[i].Data[j] * gy[i].Data[j];
                    sumdx += gx[j];
                }

                for (int j = 0; j < gx.Length; j++)
                {
                    gx[j] -= prevOutput[i].Data[j] * sumdx;
                }
                
                result[i] = new NdArray(gx, gy[i].Shape);
            }
#if !DEBUG
            );
#endif
            return result;
        }
    }
}
