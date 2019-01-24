using System;

namespace KelpNet
{
    [Serializable]
    public abstract class MultiInputFunction<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        protected Func<NdArray<T>[], NdArray<T>> MultiInputForward;
        protected Action<NdArray<T>, NdArray<T>[]> MultiOutputBackward;

        protected MultiInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T>[] Forward(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            for (int i = 0; i < xs.Length; i++)
            {
                xs[i].UseCount++;
            }

            return new[] { MultiInputForward(xs) };
        }

        public override void Backward(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

            BackwardCountUp();

            for (int i = 0; i < xs.Length; i++)
            {
                xs[i].UseCount--;
            }

            MultiOutputBackward(ys[0], xs);
        }

        public override NdArray<T>[] Predict(params NdArray<T>[] xs)
        {
            return new[] { MultiInputForward(xs) };
        }
    }
}
