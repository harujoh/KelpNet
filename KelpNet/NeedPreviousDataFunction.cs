namespace KelpNet
{
    //前回実行時の入出力を自動的に扱う
    public abstract class NeedPreviousDataFunction : PredictableFunction, IBatchable
    {
        private NdArray[] _prevInput = new NdArray[1];
        private NdArray[] _prevOutput = new NdArray[1];

        protected abstract NdArray ForwardSingle(NdArray x);
        protected abstract NdArray BackwardSingle(NdArray gy, NdArray prevInput, NdArray prevOutput);

        public override NdArray Forward(NdArray x, int batchId = 0)
        {
            //参照コピーにすることでメモリを節約
            this._prevInput[batchId] = x;
            this._prevOutput[batchId] = this.ForwardSingle(x);

            return this._prevOutput[batchId];
        }

        public override NdArray Backward(NdArray gy, int batchId = 0)
        {
            return this.BackwardSingle(gy, this._prevInput[batchId], this._prevOutput[batchId]);
        }

        //バッチ処理用の初期化関数
        public void InitBatch(int batchCount)
        {
            this._prevInput = new NdArray[batchCount];
            this._prevOutput = new NdArray[batchCount];
        }
    }
}
