using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public abstract class MultiInputFunction : Function
    {
        protected Func<NdArray[], NdArray> MultiInputForward;
        protected Action<NdArray, NdArray[]> MultiOutputBackward;

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

            BackwardCountUp();

            foreach (NdArray x in xs)
            {
                x.UseCount--;
            }

            MultiOutputBackward(ys[0], xs);
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            return new[] { MultiInputForward(xs) };
        }
    }
}
