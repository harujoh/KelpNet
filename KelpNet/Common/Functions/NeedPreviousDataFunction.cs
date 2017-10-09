using System;
using System.Collections.Generic;

namespace KelpNet.Common.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousDataFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<NdArray> _prevInput = new List<NdArray>();
        private readonly List<NdArray> _prevOutput = new List<NdArray>();

        protected Func<NdArray, NdArray> NeedPreviousForward;
        protected Func<NdArray, NdArray, NdArray, NdArray> NeedPreviousBackward;

        protected NeedPreviousDataFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
            Forward = ForwardCpu;
            Backward = BackwardCpu;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            this._prevInput.Add(x);

            NdArray prevoutput = this.NeedPreviousForward(x);
            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        public NdArray BackwardCpu(NdArray gy)
        {
            BackwardCountUp();

            NdArray prevInput = this._prevInput[this._prevInput.Count - 1];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            NdArray prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput, prevOutput);
        }

        public override NdArray Predict(NdArray x)
        {
            return this.NeedPreviousForward(x);
        }
    }
}
