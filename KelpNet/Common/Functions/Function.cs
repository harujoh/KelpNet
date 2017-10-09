using System;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Common.Functions
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function
    {
        public string Name;

        public bool GpuEnable{get; protected set;}

        public FunctionParameter[] Parameters = { };
        public Optimizer[] Optimizers;

        public readonly int OutputCount;
        public readonly int InputCount;

        public Func<NdArray, NdArray> Forward;
        public Func<NdArray, NdArray> Backward;

        //コンストラクタ
        protected Function(string name, int inputCount = 0, int oututCount = 0)
        {
            this.Name = name;
            this.InputCount = inputCount;
            this.OutputCount = oututCount;
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
            foreach (FunctionParameter parameter in this.Parameters)
            {
                parameter.CountUp();
            }
        }

        //評価関数
        public virtual NdArray Predict(NdArray input)
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

        //ある処理実行後に特定のデータを初期値に戻す処理
        public virtual void ResetState()
        {
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
