using System;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions
{
    //FunctionStackに積み上げるFunctionの基底クラス
    [Serializable]
    public abstract class Function
    {
        public string Name;

        public FunctionParameter[] Parameters = { };
        public Optimizer[] Optimizers;

        protected readonly int OutputCount;
        protected readonly int InputCount;

        //コンストラクタ
        protected Function(string name, int inputCount = 0, int oututCount = 0)
        {
            this.Name = name;
            this.InputCount = inputCount;
            this.OutputCount = oututCount;
        }

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

        public virtual BatchArray Backward(BatchArray gy)
        {
            //バッチは内部で割引を行うためgy.Lengthでの加算の必要がない
            foreach (FunctionParameter parameter in this.Parameters)
            {
                parameter.TrainCount++;
            }

            return this.BackwardSingle(gy);
        }

        //通常であれば非バッチ呼び出しを仮想とするが、
        //バッチ専用関数がスタンダードで非バッチ関数がイレギュラーであるため
        protected abstract BatchArray ForwardSingle(BatchArray x);
        protected abstract BatchArray BackwardSingle(BatchArray gy);

        //評価関数
        public virtual BatchArray Predict(BatchArray input)
        {
            return this.ForwardSingle(input);
        }

        //訓練カウントを使って各Functionの傾きを補正
        public virtual void Reduce()
        {
            foreach (FunctionParameter parameter in this.Parameters)
            {
                parameter.Reduce();
            }
        }

        public virtual void Update()
        {
            //更新実行前に訓練カウントを使って各Functionの傾きを補正
            foreach (Optimizer optimizer in this.Optimizers)
            {
                optimizer.Update();
            }
        }

        public virtual void ClearGrads()
        {
            foreach (FunctionParameter parameter in this.Parameters)
            {
                parameter.ClearGrad();
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
    }
}
