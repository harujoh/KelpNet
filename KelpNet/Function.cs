using System;
using System.Collections.Generic;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    public abstract class Function
    {
        public string Name;

        public List<OptimizeParameter> Parameters = new List<OptimizeParameter>();

        public int OutputCount;
        public int InputCount;

        protected abstract NdArray ForwardSingle(NdArray x, int batchId = 0);
        protected abstract NdArray BackwardSingle(NdArray gy, int batchId = 0);

        protected Function(string name)
        {
            this.Name = name;
        }

        public NdArray Forward(NdArray x, int batchId = 0)
        {
            return this.ForwardSingle(x, batchId);
        }

        public NdArray Backward(NdArray gy, int batchId = 0)
        {
            foreach (OptimizeParameter parameter in this.Parameters)
            {
                parameter.TrainCount++;
            }

            return this.BackwardSingle(gy, batchId);
        }

        public virtual void ResetState()
        {
        }

        //バッチ実行前に初期化が必要な関数に使用
        public virtual void InitBatch(int batchCount)
        {            
        }

        //初期値が入力されなかった場合、この関数で初期化を行う
        protected void InitWeight(NdArray array, double masterScale = 1.0)
        {
            var localScale = 1 / Math.Sqrt(2);
            var fanIn = this.GetFans(array.Shape);
            var s = localScale * Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < array.Length; i++)
            {
                array.Data[i] = this.Normal(s) * masterScale;
            }
        }

        double Normal(double scale = 0.05)
        {
            Mother.Sigma = scale;
            return Mother.RandomNormal();
        }

        int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }

        public override string ToString()
        {
            return this.Name;
        }
    }
}
