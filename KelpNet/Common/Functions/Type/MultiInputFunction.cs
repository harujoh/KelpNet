using System;

namespace KelpNet.Common.Functions.Type
{
    [Serializable]
    public abstract class MultiInputFunction : Function
    {
        protected Func<NdArray[], NdArray> MultiInputForward;
        protected Action<NdArray, NdArray[]> MultiOutputBackward;

        protected MultiInputFunction(string name) : base(name)
        {
        }

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            foreach (NdArray x in xs)
            {
                x.UseCount++;
            }

            return new []{MultiInputForward(xs)};
        }

        public override void Backward(params NdArray[] ys)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

            BackwardCountUp();

            foreach (NdArray x in xs)
            {
                x.UseCount--;
            }

            MultiOutputBackward(ys[0], xs);
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return new []{MultiInputForward(xs)};
        }
    }
}
