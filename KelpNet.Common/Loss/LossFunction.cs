using System;

namespace KelpNet
{
    public abstract class LossFunction<T, LabelType> where T : unmanaged, IComparable<T> where LabelType : unmanaged, IComparable<LabelType>
    {
        protected Func<NdArray<T>[], NdArray<LabelType>[], T> EvaluateFunc { get; set; }

        public T Evaluate(NdArray<T> input, NdArray<LabelType> teachSignal)
        {
            return EvaluateFunc(new[] { input }, new[] { teachSignal });
        }

        public T Evaluate(NdArray<T>[] input, NdArray<LabelType>[] teachSignal)
        {
            return EvaluateFunc(input, teachSignal);
        }
    }
}
