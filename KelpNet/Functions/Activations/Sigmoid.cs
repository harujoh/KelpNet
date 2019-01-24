using System;

namespace KelpNet
{
    [Serializable]
    public class Sigmoid<T> : CompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, null, name, inputNames, outputNames)
        {
        }

        internal override Real<T> ForwardActivate(Real<T> x)
        {
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        internal override Real<T> BackwardActivate(Real<T> gy, Real<T> y)
        {
            return gy * y * (1.0f - y);
        }
    }
}
