using KelpNet.Common;

namespace KelpNet
{
    public interface LossFunction
    {
        NdArray Evaluate(NdArray input, NdArray teachSignal, out double loss);
        NdArray[] Evaluate(NdArray[] input, NdArray[] teachSignal, out double loss);
    }
}
