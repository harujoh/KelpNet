namespace KelpNet
{
    public interface ICompressibleActivation : ISelectableSingleInputFunction
    {
        Real ForwardActivate(Real x);
        Real BackwardActivate(Real gy, Real y);
    }

    public static class CompressibleActivation
    {
        public static void Initialize(this ICompressibleActivation compressibleActivation)
        {
            compressibleActivation.SingleInputForward = compressibleActivation.NeedPreviousForwardCpu;
            compressibleActivation.SingleOutputBackward = compressibleActivation.NeedPreviousBackwardCpu;
        }

        public static NdArray NeedPreviousForwardCpu(this ICompressibleActivation compressibleActivation, NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = compressibleActivation.ForwardActivate(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, compressibleActivation);
        }

        public static void NeedPreviousBackwardCpu(this ICompressibleActivation compressibleActivation, NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += compressibleActivation.BackwardActivate(y.Grad[i], y.Data[i]);
            }
        }

    }
}
