namespace KelpNet
{
    public interface INeedPreviousFunction : ISelectableSingleInputFunction
    {
        NdArray NeedPreviousForwardCpu(NdArray x);
        void NeedPreviousBackwardCpu(NdArray y, NdArray x);
    }
}
