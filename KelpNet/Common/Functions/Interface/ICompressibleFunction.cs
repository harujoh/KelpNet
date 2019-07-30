namespace KelpNet
{
    public interface ICompressibleFunction : INeedPreviousFunction
    {
        ICompressibleActivation Activation { get; set; }
    }

    public static class CompressibleFunction
    {
        public static void Initialize(this ICompressibleFunction compressibleFunction, ICompressibleActivation activation)
        {
            compressibleFunction.Activation = activation;

            compressibleFunction.SingleInputForward = compressibleFunction.NeedPreviousForwardCpu;
            compressibleFunction.SingleOutputBackward = compressibleFunction.NeedPreviousBackwardCpu;
        }

        public static Real[] GetActivatedgy(this ICompressibleFunction compressibleFunction, NdArray y)
        {
            Real[] activatedgy = new Real[y.Grad.Length];

            for (int i = 0; i < activatedgy.Length; i++)
            {
                activatedgy[i] = compressibleFunction.Activation.BackwardActivate(y.Grad[i], y.Data[i]);
            }

            return activatedgy;
        }

    }
}
