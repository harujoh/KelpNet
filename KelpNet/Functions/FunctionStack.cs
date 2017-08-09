using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Functions
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack : Function
    {
        //すべての層がココにFunctionクラスとして保管される
        public readonly Function[] Functions;

        //コンストラクタ
        public FunctionStack(params Function[] functions) : base("FunctionStack")
        {
            this.Functions = functions;

            List<FunctionParameter> result = new List<FunctionParameter>();

            foreach (Function function in this.Functions)
            {
                foreach (FunctionParameter parameter in function.Parameters)
                {
                    result.Add(parameter);
                }
            }

            this.Parameters = result.ToArray();
        }

        //Functionとして呼び出された時にバトンを渡す
        protected override BatchArray ForwardSingle(BatchArray x)
        {
            return this.Forward(x);
        }

        //Functionとして呼び出された時にバトンを渡す
        protected override BatchArray BackwardSingle(BatchArray gy)
        {
            return this.Backward(gy);
        }

        //Forward
        public override BatchArray Forward(BatchArray input)
        {
            BatchArray result = this.Functions[0].Forward(input);

            for (int i = 1; i < this.Functions.Length; i++)
            {
                result = this.Functions[i].Forward(result);
            }

            return result;
        }

        //Backward
        public override BatchArray Backward(BatchArray backwardResult)
        {
            for (int i = this.Functions.Length - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            return backwardResult;
        }

        //重みの更新処理
        public override void Update()
        {
            //更新実行前に訓練カウントを使って各Functionの傾きを補正
            this.Reduce();

            foreach (Optimizer optimizer in this.Optimizers)
            {
                optimizer.Update();
            }

            this.ClearGrads();
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public override void ResetState()
        {
            foreach (Function function in this.Functions)
            {
                function.ResetState();
            }
        }

        //予想を実行する
        public override BatchArray Predict(BatchArray forwardResult)
        {
            foreach (Function function in this.Functions)
            {
                forwardResult = function.Predict(forwardResult);
            }

            return forwardResult;
        }

        //コピーを作成するメソッド
        public FunctionStack Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
