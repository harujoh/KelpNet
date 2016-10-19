using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Interface;

namespace KelpNet.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousDataFunction : Function, IPredictableFunction
    {
        //後入れ先出しリスト
        private List<NdArray[]> _prevInput = new List<NdArray[]>();
        private List<NdArray[]> _prevOutput = new List<NdArray[]>();

        protected abstract NdArray[] NeedPreviousForward(NdArray[] x);
        protected abstract NdArray[] NeedPreviousBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput);

        protected NeedPreviousDataFunction(string name) : base(name)
        {
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            this._prevInput.Add(DeepCopyHelper.DeepCopy(x));
            this._prevOutput.Add(this.NeedPreviousForward(x));

            return this._prevOutput.Last();
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            var prevInput = this._prevInput.Last();
            var prevOutput = this._prevOutput.Last();

            this._prevInput.RemoveAt(this._prevInput.Count - 1);
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput , prevOutput);
        }

        public virtual NdArray[] Predict(NdArray[] input)
        {
            return this.ForwardSingle(input);
        }
    }
}
