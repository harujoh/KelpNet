using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace KelpNet
{
    [DataContract(Name = "MultiOutputFunction", Namespace = "KelpNet")]
    public abstract class MultiOutputFunction<T> : Function<T> where T:unmanaged, IComparable<T>
    {
        protected Func<NdArray<T>, NdArray<T>[]> SingleInputForward { get; set; }

        protected Action<NdArray<T>[], NdArray<T>> MultiOutputBackward { get; set; }

        List<NdArray<T>[]> PrevOutputs = new List<NdArray<T>[]>();

        List<NdArray<T>[]> UsedPrevOutputs = new List<NdArray<T>[]>();

        protected MultiOutputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            this.Forward = ForwardMO;
            this.Backward = BackwardMO;
            this.Predict = xs => SingleInputForward(xs[0]);
        }

        public NdArray<T>[] ForwardMO(params NdArray<T>[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;
            xs[0].UseCount++;

            NdArray<T>[] result = SingleInputForward(xs[0]);
            PrevOutputs.Add(result);

            return result;
        }

        public void BackwardMO(params NdArray<T>[] ys)
        {
            NdArray<T>[] xs = PrevInputs[PrevInputs.Count - 1];

#if DEBUG
            if (xs == null || xs.Length != 1) throw new Exception("引数が正しくありません");
#endif
            xs[0].UseCount--;
            //出力した両方で使用が終わったら
            if (xs[0].UseCount <= 0)
            {
                if (xs[0].Grad == null) xs[0].InitGrad();

                InitGrad();
                BackwardCountUp();

                PrevInputs.RemoveAt(PrevInputs.Count - 1);
                NdArray<T>[] prevys = PrevOutputs[PrevOutputs.Count - 1];
                PrevOutputs.RemoveAt(PrevOutputs.Count - 1);

                UsedPrevInputs.Add(xs);
                UsedPrevOutputs.Add(prevys);

                MultiOutputBackward(prevys, xs[0]);

                if (PrevInputs.Count == 0)
                {
                    PrevInputs.AddRange(UsedPrevInputs);
                    UsedPrevInputs.Clear();
                }

                if (PrevOutputs.Count == 0)
                {
                    PrevOutputs.AddRange(UsedPrevOutputs);
                    UsedPrevOutputs.Clear();
                }
            }
        }

        public override void ResetState()
        {
            base.ResetState();

            this.PrevOutputs = new List<NdArray<T>[]>();
            this.UsedPrevOutputs = new List<NdArray<T>[]>();
        }
    }
}
