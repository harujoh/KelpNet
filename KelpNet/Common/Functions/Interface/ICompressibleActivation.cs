namespace KelpNet
{
    public interface ICompressibleActivation : ISingleInputFunction
    {
        NdArray NeedPreviousForwardCpu(NdArray x);
        void NeedPreviousBackwardCpu(NdArray y, NdArray x);
    }
}
