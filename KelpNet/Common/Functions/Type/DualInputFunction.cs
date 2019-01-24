using System;

namespace KelpNet
{
    [Serializable]
    public abstract class DualInputFunction<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        protected Func<NdArray<T>, NdArray<T>, NdArray<T>> DualInputForward;
        protected Action<NdArray<T>, NdArray<T>, NdArray<T>> DualOutputBackward;

        protected DualInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T>[] Forward(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;
            xs[1].UseCount++;

            return new[] { DualInputForward(xs[0], xs[1]) };
        }

        public override void Backward(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 2) throw new Exception("引数が正しくありません");
#endif
            BackwardCountUp();

            xs[0].UseCount--;
            xs[1].UseCount--;

            DualOutputBackward(ys[0], xs[0], xs[1]);
        }

        public override NdArray<T>[] Predict(params NdArray<T>[] xs)
        {
            return new[] { DualInputForward(xs[0], xs[1]) };
        }
    }
}
