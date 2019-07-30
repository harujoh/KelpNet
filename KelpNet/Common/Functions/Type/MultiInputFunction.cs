using System;

namespace KelpNet
{
    [Serializable]
    public abstract class MultiInputFunction : Function
    {
        protected abstract NdArray MultiInputForward(params NdArray[] xs);
        protected abstract void MultiOutputBackward(NdArray y, NdArray[] xs);

        protected MultiInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            foreach (NdArray x in xs)
            {
                x.UseCount++;
            }

            return new[] { MultiInputForward(xs) };
        }

        public override void Backward(params NdArray[] ys)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

            InitGrad();
            BackwardCountUp();

            foreach (NdArray x in xs)
            {
                x.UseCount--;
                if (x.Grad == null) x.InitGrad();
            }

            UsedPrevInputs.Add(xs);

            MultiOutputBackward(ys[0], xs);

            //使い切ったら復活
            if (PrevInputs.Count == 0)
            {
                PrevInputs.AddRange(UsedPrevInputs);
                UsedPrevInputs.Clear();
            }
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return new[] { MultiInputForward(xs) };
        }
    }
}
