using System;
using System.Runtime.Serialization;

namespace KelpNet
{
    [DataContract(Name = "SingleInputFunction", Namespace = "KelpNet")]
    public abstract class SingleInputFunction<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        public Func<NdArray<T>, NdArray<T>> SingleInputForward { get; set; }
        public Action<NdArray<T>, NdArray<T>> SingleOutputBackward { get; set; }

        protected SingleInputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.Forward = this.ForwardBase;
            base.Backward = this.BackwardBase;
            this.Predict = xs => new[] { SingleInputForward(xs[0]) };
        }

        public NdArray<T>[] ForwardBase(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;

            return new[] { SingleInputForward(xs[0]) };
        }

        public void BackwardBase(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 1) throw new Exception("引数が正しくありません");
#endif
            InitGrad();
            BackwardCountUp();

            xs[0].UseCount--;
            if (xs[0].Grad == null) xs[0].InitGrad();

            UsedPrevInputs.Add(xs);

            SingleOutputBackward(ys[0], xs[0]);

            //使い切ったら復活
            if (PrevInputs.Count == 0)
            {
                PrevInputs.AddRange(UsedPrevInputs);
                UsedPrevInputs.Clear();
            }
        }
    }
}
