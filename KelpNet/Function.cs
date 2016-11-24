using System;
using System.Collections.Generic;
using KelpNet.Common;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function
    {
        public string Name;

        public List<OptimizeParameter> Parameters = new List<OptimizeParameter>();

        protected readonly int OutputCount;
        protected readonly int InputCount;

        //コンストラクタ
        protected Function(string name, int inputCount = 0, int oututCount = 0)
        {
            this.Name = name;
            this.InputCount = inputCount;
            this.OutputCount = oututCount;
        }

        //外部公開用
        public virtual NdArray[] Forward(NdArray[] x)
        {
            return this.ForwardSingle(x);
        }

        public virtual NdArray[] Backward(NdArray[] gy)
        {
            //バッチは内部で割引を行うためgy.Lengthでの加算の必要がない
            for (int i = 0; i < this.Parameters.Count; i++)
            {
                this.Parameters[i].TrainCount++;
            }

            return this.BackwardSingle(gy);
        }

        //通常であれば非バッチ呼び出しを仮想とするが、
        //バッチ専用関数がスタンダードで非バッチ関数がイレギュラーであるため
        protected abstract NdArray[] ForwardSingle(NdArray[] x);
        protected abstract NdArray[] BackwardSingle(NdArray[] gy);

        //外部公開用非バッチ関数
        public virtual NdArray Forward(NdArray x)
        {
            return this.ForwardSingle(x);
        }

        public virtual NdArray Backward(NdArray gy)
        {
            for (int i = 0; i < this.Parameters.Count; i++)
            {
                this.Parameters[i].TrainCount++;
            }

            return this.BackwardSingle(gy);
        }

        //任意で個別に非バッチ関数が書けるように用意
        protected virtual NdArray ForwardSingle(NdArray x)
        {
            return this.ForwardSingle(new[] { x })[0];
        }

        protected virtual NdArray BackwardSingle(NdArray gy)
        {
            return this.BackwardSingle(new[] { gy })[0];
        }

        //評価関数
        public virtual NdArray[] Predict(NdArray[] input)
        {
            return this.ForwardSingle(input);
        }

        public virtual NdArray Predict(NdArray input)
        {
            return this.ForwardSingle(input);
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public virtual void ResetState()
        {
        }

        //名前を返す
        public override string ToString()
        {
            return this.Name;
        }

        //初期値が入力されなかった場合、この関数で初期化を行う
        protected void InitWeight(NdArray array, double masterScale = 1.0)
        {
            double localScale = 1 / Math.Sqrt(2);
            int fanIn = this.GetFans(array.Shape);
            double s = localScale * Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < array.Length; i++)
            {
                array.Data[i] = this.Normal(s) * masterScale;
            }
        }

        private double Normal(double scale = 0.05)
        {
            Mother.Sigma = scale;
            return Mother.RandomNormal();
        }

        private int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }
    }
}
