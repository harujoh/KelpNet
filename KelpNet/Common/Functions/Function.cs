using System;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function
    {
        public string Name;

        public bool IsGpu { get; private set; }

        public FunctionParameter[] Parameters = { };
        public Optimizer[] Optimizers;

        public readonly int OutputCount;
        public readonly int InputCount;

        //コンストラクタ
        protected Function(string name, int inputCount = 0, int oututCount = 0)
        {
            this.Name = name;
            this.InputCount = inputCount;
            this.OutputCount = oututCount;
        }

        protected abstract BatchArray ForwardSingle(BatchArray x);
        protected abstract BatchArray BackwardSingle(BatchArray gy);

        public void SetOptimizer(params Optimizer[] optimizers)
        {
            this.Optimizers = optimizers;

            foreach (Optimizer optimizer in optimizers)
            {
                optimizer.AddFunctionParameters(this.Parameters);
            }
        }

        //外部公開用
        public virtual BatchArray Forward(BatchArray x)
        {
            return this.ForwardSingle(x);
        }

        //外部公開用
        public virtual BatchArray Backward(BatchArray gy)
        {
            foreach (FunctionParameter parameter in this.Parameters)
            {
                parameter.TrainCount++;
            }

            return this.BackwardSingle(gy);
        }

        //評価関数
        public virtual BatchArray Predict(BatchArray input)
        {
            return this.ForwardSingle(input);
        }

        public virtual void Update()
        {
            //更新実行前に訓練カウントを使って各Functionの傾きを補正
            foreach (Optimizer optimizer in this.Optimizers)
            {
                optimizer.Update();
            }
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public virtual void ResetState()
        {
        }

        public bool SetUpGpu()
        {
            this.IsGpu = Weaver.Enable;
            if (this.IsGpu)
            {
                CreateKernel();
            }

            return this.IsGpu;
        }

        protected virtual void CreateKernel()
        {            
        }

        //名前を返す
        public override string ToString()
        {
            return this.Name;
        }
    }
}
