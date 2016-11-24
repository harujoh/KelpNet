using System;
using System.Collections.Generic;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : Function
    {
        private readonly double dropoutRatio;
        private readonly List<double[]> MaskStack = new List<double[]>();

        public Dropout(double dropoutRatio = 0.5, string name = "Dropout") : base(name)
        {
            this.dropoutRatio = dropoutRatio;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] result = new NdArray[x.Length];
            double[] mask = new double[x[0].Length];
            double scale = 1.0 / (1.0 - this.dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

#if DEBUG
            for (int i = 0; i < x.Length; i++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                double[] y = new double[x[i].Length];

                for (int j = 0; j < mask.Length; j++)
                {
                    y[j] = x[i].Data[j] * mask[j];
                }

                result[i] = new NdArray(y, x[i].Shape);
            }
#if !DEBUG
            );
#endif
            this.MaskStack.Add(mask);

            return result;
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            NdArray[] result = new NdArray[gy.Length];

            double[] mask = this.MaskStack[this.MaskStack.Count-1];
            this.MaskStack.RemoveAt(this.MaskStack.Count - 1);

#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                double[] gx = new double[gy[i].Length];

                for (int j = 0; j < mask.Length; j++)
                {
                    gx[j] = gy[i].Data[j] * mask[j];
                }

                result[i] = new NdArray(gx, gy[i].Shape);
            }
#if !DEBUG
            );
#endif

            return result;
        }

        //Predict時に何もしない
        public override NdArray[] Predict(NdArray[] input)
        {
            return input;
        }

    }
}
