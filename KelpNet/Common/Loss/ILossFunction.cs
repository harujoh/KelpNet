namespace KelpNet.Common.Loss
{
    public interface ILossFunction
    {
        BatchArray Evaluate(BatchArray input, BatchArray teachSignal, out double loss);
    }
}
