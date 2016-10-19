using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Activations
{
    [Serializable]
    public class ELU : NeedPreviousDataFunction
    {
        private readonly double _alpha;

        public ELU(double alpha = 1.0, string name = "ELU", int batchCount = 1) : base(name)
        {
            this._alpha = alpha;
        }

        protected override NdArray[] NeedPreviousForward(NdArray[] x)
        {
            NdArray[] result = new NdArray[x.Length];

#if DEBUG
            for(int i = 0; i < x.Length; i ++)
#else
            Parallel.For(0, x.Length, i => 
#endif
            { 
                double[] y = new double[x[i].Length];

                for (int j = 0; j < x[i].Length; j++)
                {
                    y[j] = x[i].Data[j] >= 0 ? x[i].Data[j] : this._alpha * (Math.Exp(x[i].Data[j]) - 1);
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
                    gx[j] = prevOutput[i].Data[j] >= 0
                        ? gy[i].Data[j]
                        : gy[i].Data[j] * this._alpha * Math.Exp(prevInput[i].Data[j]);
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
