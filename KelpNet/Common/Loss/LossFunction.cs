using System;

namespace KelpNet
{
    public abstract class LossFunction<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> Evaluate(NdArray<T> input, NdArray<T> teachSignal)
        {
            return Evaluate(new[] { input }, new[] { teachSignal });
        }

        public Real<T> Evaluate(NdArray<T>[] input, NdArray<T> teachSignal)
        {
            return Evaluate(input, new[] { teachSignal });
        }

        public abstract Real<T> Evaluate(NdArray<T>[] input, NdArray<T>[] teachSignal);
    }
}
