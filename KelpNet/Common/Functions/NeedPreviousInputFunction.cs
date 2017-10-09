using System;
using System.Collections.Generic;

namespace KelpNet.Common.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousInputFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<NdArray> _prevInput = new List<NdArray>();

        protected Func<NdArray, NdArray> NeedPreviousForward;
        protected Func<NdArray, NdArray, NdArray> NeedPreviousBackward;

        protected NeedPreviousInputFunction(string name) : base(name)
        {
            Forward = ForwardCpu;
            Backward = BackwardCpu;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            this._prevInput.Add(x);

            return this.NeedPreviousForward(x);
        }

        public NdArray BackwardCpu(NdArray gy)
        {
            BackwardCountUp();

            NdArray prevInput = this._prevInput[this._prevInput.Count - 1];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput);
        }

        public override NdArray Predict(NdArray x)
        {
            return this.NeedPreviousForward(x);
        }
    }
}
