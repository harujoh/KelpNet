using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Interface;

namespace KelpNet.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousDataFunction : Function, IPredictableFunction
    {
        //後入れ先出しリスト
        private Stack<NdArray>[] _prevInput = {new Stack<NdArray>() };
        private Stack<NdArray>[] _prevOutput = { new Stack<NdArray>() };

        protected abstract NdArray NeedPreviousForward(NdArray x);
        protected abstract NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput);

        protected NeedPreviousDataFunction(string name):base(name)
        {
        }

        protected override NdArray ForwardSingle(NdArray x, int batchId = 0)
        {
            this._prevInput[batchId].Push(new NdArray(x));
            this._prevOutput[batchId].Push(new NdArray(this.NeedPreviousForward(this._prevInput[batchId].Peek())));

            return this._prevOutput[batchId].Peek();
        }

        protected override NdArray BackwardSingle(NdArray gy, int batchId = 0)
        {
            return this.NeedPreviousBackward(gy, this._prevInput[batchId].Pop(), this._prevOutput[batchId].Pop());
        }

        //バッチ処理用の初期化関数
        public override void InitBatch(int batchCount)
        {
            this._prevInput = new Stack<NdArray>[batchCount];
            this._prevOutput = new Stack<NdArray>[batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                this._prevInput[i] = new Stack<NdArray>();
                this._prevOutput[i] = new Stack<NdArray>();
            }
        }

        public virtual NdArray Predict(NdArray input,int batchID)
        {
            return this.ForwardSingle(input, batchID);
        }
    }
}
