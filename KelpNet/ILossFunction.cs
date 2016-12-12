using KelpNet.Common;

namespace KelpNet
{
    public interface ILossFunction
    {
        NdArray Evaluate(NdArray input, NdArray teachSignal, out double loss);
        NdArray[] Evaluate(NdArray[] input, NdArray[] teachSignal, out double loss);
    }
}
