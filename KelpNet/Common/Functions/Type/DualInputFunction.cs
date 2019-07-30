using System;

namespace KelpNet
{
    [Serializable]
    public abstract class DualInputFunction : Function
    {
        protected abstract NdArray DualInputForward(NdArray a, NdArray b);
        protected abstract void DualOutputBackward(NdArray y, NdArray a, NdArray b);

        protected DualInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;
            xs[1].UseCount++;

            return new[] { DualInputForward(xs[0], xs[1]) };
        }

        public override void Backward(params NdArray[] ys)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 2) throw new Exception("引数が正しくありません");
#endif
            InitGrad();
            BackwardCountUp();

            xs[0].UseCount--;
            xs[1].UseCount--;

            if (xs[0].Grad == null) xs[0].InitGrad();
            if (xs[1].Grad == null) xs[1].InitGrad();

            UsedPrevInputs.Add(xs);

            DualOutputBackward(ys[0], xs[0], xs[1]);

            if (PrevInputs.Count == 0)
            {
                PrevInputs.AddRange(UsedPrevInputs);
                UsedPrevInputs.Clear();
            }
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return new[] { DualInputForward(xs[0], xs[1]) };
        }
    }
}
