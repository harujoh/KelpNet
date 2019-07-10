namespace KelpNet
{
    public interface ICompressibleActivation : ISingleInputFunction
    {
        Real ForwardActivate(Real x);
        Real BackwardActivate(Real gy, Real y);

        NdArray NeedPreviousForwardCpu(NdArray x);
        void NeedPreviousBackwardCpu(NdArray y, NdArray x);
    }
}
