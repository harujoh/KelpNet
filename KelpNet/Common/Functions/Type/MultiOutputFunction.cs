using System;

namespace KelpNet
{
    [Serializable]
    public abstract class MultiOutputFunction<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        protected Func<NdArray<T>, NdArray<T>[]> SingleInputForward;
        protected Action<NdArray<T>[], NdArray<T>> SingleOutputBackward;

        protected MultiOutputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray<T>[] Forward(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;

            return SingleInputForward(xs[0]);
        }

        public override void Backward(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 1) throw new Exception("引数が正しくありません");
#endif
            BackwardCountUp();

            xs[0].UseCount--;

            SingleOutputBackward(ys, xs[0]);
        }

        public override NdArray<T>[] Predict(params NdArray<T>[] xs)
        {
            return SingleInputForward(xs[0]);
        }
    }
}
