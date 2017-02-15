using System;
using System.Collections.Generic;

namespace KelpNet.Common.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousOutputFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<BatchArray> _prevOutput = new List<BatchArray>();

        protected abstract BatchArray NeedPreviousForward(BatchArray x);
        protected abstract BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevOutput);

        protected NeedPreviousOutputFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
        }

        protected override BatchArray ForwardSingle(BatchArray x)
        {
            BatchArray prevoutput = this.NeedPreviousForward(x);

            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        protected override BatchArray BackwardSingle(BatchArray gy)
        {
            BatchArray prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevOutput);
        }

        public override BatchArray Predict(BatchArray x)
        {
            return this.NeedPreviousForward(x);
        }
    }
}
