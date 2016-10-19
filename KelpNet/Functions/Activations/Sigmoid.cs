using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class Sigmoid : NeedPreviousDataFunction
    {
        public Sigmoid(string name = "Sigmoid") : base(name)
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

                for (int j = 0; j < x[i].Length; j++)
                {
                    y[j] = 1 / (1 + Math.Exp(-x[i].Data[j]));
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

                for (int j = 0; j < gy[i].Length; j++)
                {
                    gx[j] = gy[i].Data[j] * prevOutput[i].Data[j] * (1 - prevOutput[i].Data[j]);
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
