using System;
using KelpNet.Common;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : Function
    {
        private readonly double dropoutRatio;
        private double[] mask;

        public Dropout(double dropoutRatio = 0.5, string name = "Dropout") : base(name)
        {
            this.dropoutRatio = dropoutRatio;
        }

        protected override NdArray ForwardSingle(NdArray x, int batchID = -1)
        {
            double[] resultData = new double[x.Length];

            if (this.mask == null || batchID == -1)
            {
                double scale = 1.0 / (1.0 - this.dropoutRatio);

                this.mask = new double[x.Length];

                for (int i = 0; i < this.mask.Length; i++)
                {
                    this.mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
                    resultData[i] = x.Data[i] * this.mask[i];
                }
            }
            else
            {
                for (int i = 0; i < this.mask.Length; i++)
                {
                    resultData[i] = x.Data[i] * this.mask[i];
                }
            }

            return new NdArray(resultData, x.Shape);
        }

        protected override NdArray BackwardSingle(NdArray gy, int batchID = 0)
        {
            NdArray result = NdArray.ZerosLike(gy);

            for (int i = 0; i < this.mask.Length; i++)
            {
                result.Data[i] = gy.Data[i] * this.mask[i];
            }

            return result;
        }

        public override void InitBatch(int batchCount)
        {
            this.mask = null;
        }
    }
}
