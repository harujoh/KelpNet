using System;
using System.Runtime.Serialization;

namespace KelpNet
{
    [DataContract(Name = "MultiInputFunction", Namespace = "KelpNet")]
    public abstract class MultiInputFunction<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        public FunctionOptional<T> MultiInputForward { get; set; }

        public Action<NdArray<T>, NdArray<T>[]> MultiOutputBackward { get; set; }

        protected MultiInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            this.Forward = ForwardMI;
            this.Backward = BackwardMI;
            this.Predict = xs => MultiInputForward(xs);
        }

        public NdArray<T>[] ForwardMI(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            foreach (NdArray<T> x in xs)
            {
                x.UseCount++;
            }

            return MultiInputForward(xs);
        }

        public void BackwardMI(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

            InitGrad();
            BackwardCountUp();

            foreach (NdArray<T> x in xs)
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
    }
}
