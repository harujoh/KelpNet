using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class LeakyReLU : NeedPreviousDataFunction
    {
        private readonly double _slope;

        public LeakyReLU(double slope = 0.2, string name = "LeakyReLU") : base(name)
        {
            this._slope = slope;
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
                    if (y[j] < 0) y[j] *= this._slope;
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

                for (int j = 0; j < gx.Length; j++)
                {
                    gx[j] = prevOutput[i].Data[j] > 0 ? gy[i].Data[j] : prevOutput[i].Data[j] * this._slope;
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
