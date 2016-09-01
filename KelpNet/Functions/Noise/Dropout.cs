namespace KelpNet.Functions.Noise
{
    public class Dropout:Function
    {
        private readonly double dropoutRatio;
        private double[] mask;

        public Dropout(double dropoutRatio = 0.5)
        {
            this.dropoutRatio = dropoutRatio;
        }

        public override NdArray Forward(NdArray x)
        {
            NdArray result = NdArray.EmptyLike(x);
            double scale = 1.0 / (1.0 - this.dropoutRatio);
            this.mask = new double[x.Length];

            for (int i = 0; i < this.mask.Length; i++)
            {
                this.mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
                result.Data[i] = x.Data[i] * this.mask[i];
            }

            return result;
        }

        public override NdArray Backward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.EmptyLike(gy);

            for (int i = 0; i < this.mask.Length; i++)
            {
                result.Data[i] = gy.Data[i] * this.mask[i];
            }

            return result;
        }
    }
}
