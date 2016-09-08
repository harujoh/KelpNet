namespace KelpNet.Functions.Activations
{
    public class LeakyReLU : PredictableFunction
    {
        private readonly double slope;

        public LeakyReLU(double slope = 0.2)
        {
            this.slope = slope;
        }

        public override NdArray Forward(NdArray x, int batchID = 0)
        {
            NdArray y = new NdArray(x);

            for (int i = 0; i < x.Length; i++)
            {
                if (y.Data[i] < 0) y.Data[i] *= this.slope;
            }

            return y;
        }

        public override NdArray Backward(NdArray gy, NdArray prevInput, NdArray prevOutput, int batchID = 0)
        {
            NdArray result = new NdArray(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                if (prevOutput.Data[i] < 0) prevOutput.Data[i] *= this.slope;
            }

            return result;
        }
    }
}
