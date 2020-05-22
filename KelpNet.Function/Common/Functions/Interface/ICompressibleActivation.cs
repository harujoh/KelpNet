using KelpNet.CPU;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public static class ICompressibleActivationF
#else
    public static class ICompressibleActivationD
#endif
    {
        public static NdArray<Real> NeedPreviousForwardCpu(this ICompressibleActivation<Real> compressibleActivation, NdArray<Real> x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = compressibleActivation.ForwardActivate(x.Data[i]);
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, compressibleActivation);
        }

        public static void NeedPreviousBackwardCpu(this ICompressibleActivation<Real> compressibleActivation, NdArray<Real> y, NdArray<Real> x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += compressibleActivation.BackwardActivate(y.Grad[i], y.Data[i], x.Data[i]);
            }
        }

        public static Real[] GetActivatedgy(this ICompressibleActivation<Real> compressibleActivation, NdArray<Real> y, NdArray<Real> x)
        {
            Real[] activatedgy = new Real[y.Grad.Length];

            for (int i = 0; i < activatedgy.Length; i++)
            {
                activatedgy[i] = compressibleActivation.BackwardActivate(y.Grad[i], y.Data[i], x.Data[i]);
            }

            return activatedgy;
        }
    }
}
