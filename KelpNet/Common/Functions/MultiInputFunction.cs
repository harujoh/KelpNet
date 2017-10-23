using System;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class MultiInputFunction : Function
    {
        protected MultiInputFunction(string name) : base(name)
        {
        }

        protected Func<NdArray[], NdArray> MultiInputForward;
        protected Action<NdArray, NdArray[]> MultiOutputBackward;

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            foreach (NdArray x in xs)
            {
                x.UseCount++;
            }

            return new []{MultiInputForward(xs)};
        }

        public override void Backward(params NdArray[] y)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 2) throw new Exception("引数が正しくありません");
#endif
            BackwardCountUp();

            foreach (NdArray x in xs)
            {
                x.UseCount--;
            }

            MultiOutputBackward(y[0], xs);
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return new []{MultiInputForward(xs)};
        }
    }
}
