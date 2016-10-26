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

        public int OutputCount;
        public int InputCount;

        //コンストラクタ
        protected Function(string name)
        {
            this.Name = name;
        }

        public NdArray[] Forward(NdArray[] x)
        {
            return this.ForwardSingle(x);
        }

        public NdArray[] Backward(NdArray[] gy)
        {
            foreach (OptimizeParameter parameter in this.Parameters)
            {
                parameter.TrainCount += gy.Length;
            }

            return this.BackwardSingle(gy);
        }

        //通常であれば非バッチ呼び出しを仮想とすべきだが、
        //バッチ専用関数がデフォルトで非バッチ関数がイレギュラーであるため
        protected abstract NdArray[] ForwardSingle(NdArray[] x);
        protected abstract NdArray[] BackwardSingle(NdArray[] gy);

        public NdArray Forward(NdArray x)
        {
            return this.ForwardSingle(x);
        }

        public NdArray Backward(NdArray gy)
        {
            foreach (OptimizeParameter parameter in this.Parameters)
            {
                parameter.TrainCount++;
            }

            return this.BackwardSingle(gy);
        }

        //個別に非バッチ処理が書けるように用意
        public virtual NdArray ForwardSingle(NdArray x)
        {
            return this.ForwardSingle(new [] {x})[0];
        }

        public virtual NdArray BackwardSingle(NdArray gy)
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

        public override string ToString()
        {
            return this.Name;
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
