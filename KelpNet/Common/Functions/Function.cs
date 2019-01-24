using System;
using System.Collections.Generic;
using System.Linq;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function<T> where T : unmanaged, IComparable<T>
    {
        public string Name;

        public NdArray<T>[] Parameters = { };
        public Optimizer<T>[] Optimizers = { };

        [NonSerialized]
        public List<NdArray<T>[]> PrevInputs = new List<NdArray<T>[]>();

        public abstract NdArray<T>[] Forward(params NdArray<T>[] xs);
        public virtual void Backward(params NdArray<T>[] ys){}

        public string[] InputNames;
        public string[] OutputNames;

        //コンストラクタ
        protected Function(string name, string[] inputNames = null, string[] outputNames = null)
        {
            this.Name = name;

            if (inputNames != null)
            {
                this.InputNames = inputNames.ToArray();
            }

            if (outputNames != null)
            {
                this.OutputNames = outputNames.ToArray();
            }
        }

        public virtual void SetOptimizer(params Optimizer<T>[] optimizers)
        {
            this.Optimizers = optimizers;

            foreach (Optimizer<T> optimizer in optimizers)
            {
                optimizer.AddFunctionParameters(this.Parameters);
            }
        }

        //パラメータを更新する時に呼ぶ関数
        protected void BackwardCountUp()
        {
            foreach (NdArray<T> parameter in this.Parameters)
            {
                parameter.CountUp();
            }
        }

        //評価関数
        public virtual NdArray<T>[] Predict(params NdArray<T>[] input)
        {
            return this.Forward(input);
        }

        public virtual void Update()
        {
            foreach (Optimizer<T> optimizer in this.Optimizers)
            {
                optimizer.Update();
            }
        }

        //RNN等で使い切れなかった入力データを初期化
        public virtual void ResetState()
        {
            this.PrevInputs = new List<NdArray<T>[]>();
        }

        //名前を返す
        public override string ToString()
        {
            return this.Name;
        }
    }
}
