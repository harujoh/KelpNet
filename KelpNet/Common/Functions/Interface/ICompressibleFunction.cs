namespace KelpNet
{
    public interface ICompressibleFunction : ISingleInputFunction
    {
        ICompressibleActivation Activator { get; set; }

        NdArray NeedPreviousForwardCpu(NdArray input);
        void NeedPreviousBackwardCpu(NdArray y, NdArray x);
    }
}
