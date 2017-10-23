namespace KelpNet.Common.Loss
{
    public abstract class LossFunction
    {
        public Real Evaluate(NdArray input, NdArray teachSignal)
        {
            return Evaluate(new[] { input }, new[] { teachSignal });
        }

        public Real Evaluate(NdArray[] input, NdArray teachSignal)
        {
            return Evaluate(input, new[] { teachSignal });
        }

        public abstract Real Evaluate(NdArray[] input, NdArray[] teachSignal);
    }
}
