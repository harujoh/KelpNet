namespace KelpNet
{
    public interface ICompressibleFunction : ISingleInputFunction
    {
        CompressibleActivation Activator { get; set; }

        NdArray NeedPreviousForwardCpu(NdArray input);
        void NeedPreviousBackwardCpu(NdArray y, NdArray x);
    }
}
