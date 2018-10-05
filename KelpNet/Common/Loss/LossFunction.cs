#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    public abstract class LossFunction
    {
        public Real Evaluate(NdArray input, NdArray teachSignal)
        {
            return Evaluate(new[] { input }, new[] { teachSignal });
        }

        public Real Evaluate(NdArray[] input, NdArray teachSignal)
        {
            return Evaluate(input, new[] { teachSignal });
        }

        public abstract Real Evaluate(NdArray[] input, NdArray[] teachSignal);
    }
}
