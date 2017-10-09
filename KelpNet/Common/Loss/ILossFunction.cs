namespace KelpNet.Common.Loss
{
    public interface ILossFunction
    {
        NdArray Evaluate(NdArray input, NdArray teachSignal, out Real loss);
    }
}
