namespace KelpNet
{
    public interface ICompressibleFunction : ISingleInputFunction
    {
        ICompressibleActivation Activation { get; set; }
    }

    public static class CompressibleFunction
    {
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
