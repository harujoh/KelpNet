using System;
using System.Runtime.Serialization;

namespace KelpNet
{
    //第一引数がconst値のときに使用。主にBasicMathで使用する。
    [DataContract(Name = "DualInputFunction", Namespace = "KelpNet")]
    public abstract class NdArrayAndConstFunction<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        public abstract NdArray<T> DualInputForward(NdArray<T> a, T b);
        public abstract void DualOutputBackward(NdArray<T> y, NdArray<T> a, T b);

        protected NdArrayAndConstFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            this.Forward = ForwardDI;
            this.Backward = BackwardDI;
            this.Predict = xs => new[] { DualInputForward(xs[0], xs[1].Data[0]) };
        }

        public NdArray<T>[] ForwardDI(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;

            return new[] { DualInputForward(xs[0], xs[1].Data[0]) };
        }

        public void BackwardDI(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 2) throw new Exception("引数が正しくありません");
#endif
            InitGrad();
            BackwardCountUp();

            xs[0].UseCount--;

            if (xs[0].Grad == null) xs[0].InitGrad();

            UsedPrevInputs.Add(xs);

            DualOutputBackward(ys[0], xs[0], xs[1].Data[0]);

            if (PrevInputs.Count == 0)
            {
                PrevInputs.AddRange(UsedPrevInputs);
                UsedPrevInputs.Clear();
            }
        }
    }
}
