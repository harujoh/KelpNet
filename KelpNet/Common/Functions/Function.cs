using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Common.Functions
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function
    {
        public string Name;

        public bool GpuEnable { get; protected set; }

        public NdArray[] Parameters = { };
        public Optimizer[] Optimizers = { };

        [NonSerialized]
        public List<NdArray[]> PrevInputs = new List<NdArray[]>();

        public abstract NdArray[] Forward(params NdArray[] xs);
        public virtual void Backward(params NdArray[] ys){}

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

        public virtual void SetOptimizer(params Optimizer[] optimizers)
        {
            this.Optimizers = optimizers;

            foreach (Optimizer optimizer in optimizers)
            {
                optimizer.AddFunctionParameters(this.Parameters);
            }
        }

        //パラメータを更新する時に呼ぶ関数
        protected void BackwardCountUp()
        {
            foreach (NdArray parameter in this.Parameters)
            {
                parameter.CountUp();
            }
        }

        //評価関数
        public virtual NdArray[] Predict(params NdArray[] input)
        {
            return this.Forward(input);
        }

        public virtual void Update()
        {
            foreach (Optimizer optimizer in this.Optimizers)
            {
                optimizer.Update();
            }
        }

        //RNN等で使い切れなかった入力データを初期化
        public virtual void ResetState()
        {
            this.PrevInputs = new List<NdArray[]>();
        }

        //名前を返す
        public override string ToString()
        {
            return this.Name;
        }

        //コピーを作成するメソッド
        public Function Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
