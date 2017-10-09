using System;
using System.Collections.Generic;

namespace KelpNet.Common.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousOutputFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<NdArray> _prevOutput = new List<NdArray>();

        protected Func<NdArray, NdArray> NeedPreviousForward;
        protected Func<NdArray, NdArray, NdArray> NeedPreviousBackward;

        protected NeedPreviousOutputFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
            Forward = ForwardCpu;
            Backward = BackwardCpu;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            NdArray prevoutput = this.NeedPreviousForward(x);

            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        public NdArray BackwardCpu(NdArray gy)
        {
            BackwardCountUp();

            NdArray prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevOutput);
        }

        public override NdArray Predict(NdArray x)
        {
            return this.NeedPreviousForward(x);
        }
    }
}
