namespace KelpNet.Functions.Noise
{
    public class Dropout : Function,IBatchable
    {
        private readonly double dropoutRatio;
        private double[] mask;

        public Dropout(double dropoutRatio = 0.5)
        {
            this.dropoutRatio = dropoutRatio;
        }

        public override NdArray Forward(NdArray x, int batchID = -1)
        {
            NdArray result = NdArray.EmptyLike(x);

            if (this.mask == null || batchID == -1)
            {
                double scale = 1.0 / (1.0 - this.dropoutRatio);

                this.mask = new double[x.Length];

                for (int i = 0; i < this.mask.Length; i++)
                {
                    this.mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
                    result.Data[i] = x.Data[i]*this.mask[i];
                }
            }
            else
            {
                for (int i = 0; i < this.mask.Length; i++)
                {
                    result.Data[i] = x.Data[i] * this.mask[i];
                }
            }

            return result;
        }

        public override NdArray Backward(NdArray gy, int batchID = 0)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < this.mask.Length; i++)
            {
                result.Data[i] = gy.Data[i] * this.mask[i];
            }

            return result;
        }

        public void InitBatch(int batchCount)
        {
            this.mask = null;
        }
    }
}
