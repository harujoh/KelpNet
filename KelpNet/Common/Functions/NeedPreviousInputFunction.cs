using System;
using System.Collections.Generic;

namespace KelpNet.Common.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousInputFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<BatchArray> _prevInput = new List<BatchArray>();

        protected abstract BatchArray NeedPreviousForward(BatchArray x, bool isGpu);
        protected abstract BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput, bool isGpu);

        protected NeedPreviousInputFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
        }

        protected override BatchArray ForwardSingle(BatchArray x, bool isGpu)
        {
            this._prevInput.Add(x);

            return this.NeedPreviousForward(x, isGpu);
        }

        protected override BatchArray BackwardSingle(BatchArray gy, bool isGpu)
        {
            BatchArray prevInput = this._prevInput[this._prevInput.Count - 1];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput, isGpu);
        }

        public override BatchArray Predict(BatchArray x, bool isGpu = true)
        {
            return this.NeedPreviousForward(x, isGpu);
        }
    }
}
