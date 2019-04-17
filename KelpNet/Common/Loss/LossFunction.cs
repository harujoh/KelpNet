namespace KelpNet
{
    public abstract class LossFunction
    {
        public Real Evaluate(NdArray input, NdArray teachSignal)
        {
            return Evaluate(new[] { input }, new[] { teachSignal });
        }

        public abstract Real Evaluate(NdArray[] input, NdArray[] teachSignal);
    }
}
