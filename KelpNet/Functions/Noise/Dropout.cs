using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : Function
    {
        private readonly double dropoutRatio;
        private readonly List<double[]> maskStack = new List<double[]>();

        public Dropout(double dropoutRatio = 0.5, string name = "Dropout") : base(name)
        {
            this.dropoutRatio = dropoutRatio;
        }

        protected override BatchArray ForwardSingle(BatchArray x)
        {
            //BatchArray result = new NdArray[x.Length];
            double[] result = new double[x.Data.Length];
            double[] mask = new double[x.Length];
            double scale = 1.0 / (1.0 - this.dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[i + b * x.Length] = x.Data[i + b * x.Length] * mask[i];
                }

                //result[b] = NdArray.Convert(result, x.Shape);
            }

            this.maskStack.Add(mask);

            return BatchArray.Convert(result, x.Shape, x.BatchCount);
        }

        protected override BatchArray BackwardSingle(BatchArray gy)
        {
            double[] result = new double[gy.Data.Length];
            double[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            for (int b = 0; b < gy.BatchCount; b++)
            {
                for (int j = 0; j < mask.Length; j++)
                {
                    result[j + b * gy.Length] = gy.Data[j + b * gy.Length] * mask[j];
                }
            }

            return BatchArray.Convert(result, gy.Shape, gy.BatchCount);
        }

        //Predict時に何もしない
        public override BatchArray Predict(BatchArray input)
        {
            return input;
        }

    }
}
