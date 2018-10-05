using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public class Tanh : CompressibleActivation
    {
        const string FUNCTION_NAME = "Tanh";

        public Tanh(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, null, name, inputNames, outputNames)
        {
        }

        internal override Real ForwardActivate(Real x)
        {
            return (Real)Math.Tanh(x);
        }

        internal override Real BackwardActivate(Real gy, Real y)
        {
            return gy * (1.0f - y * y);
        }
    }
}
