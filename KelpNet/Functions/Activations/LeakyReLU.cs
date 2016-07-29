namespace KelpNet.Functions.Activations
{
    public class LeakyReLU : Function, IPredictableFunction
    {
        private readonly double slope;

        public LeakyReLU(double slope = 0.2)
        {
            this.slope = slope;
        }

        public override NdArray Forward(NdArray x)
        {
            PrevOutput = new NdArray(x);

            for (int i = 0; i < x.Length; i++)
            {
                if (PrevOutput.Data[i] < 0) PrevOutput.Data[i] *= this.slope;
            }

            return PrevOutput;
        }

        public override NdArray Backward(NdArray gy)
        {
            NdArray result = new NdArray(gy);

            for (int i = 0; i < gy.Length; i++)
            {
                if (PrevOutput.Data[i] < 0) PrevOutput.Data[i] *= this.slope;
            }

            return result;
        }

        public NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}
