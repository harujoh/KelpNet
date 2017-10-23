using System;

namespace KelpNet.Common.Functions
{
    [Serializable]
    public abstract class MultiOutputFunction : Function
    {
        protected Func<NdArray, NdArray[]> SingleInputForward;
        protected Action<NdArray[], NdArray> SingleOutputBackward;

        protected MultiOutputFunction(string name) : base(name)
        {
        }

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;

            return SingleInputForward(xs[0]);
        }

        public override void Backward(params NdArray[] y)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 1) throw new Exception("引数が正しくありません");
#endif
            BackwardCountUp();

            xs[0].UseCount--;

            SingleOutputBackward(y, xs[0]);
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return SingleInputForward(xs[0]);
        }
    }
}
