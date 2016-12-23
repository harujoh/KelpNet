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
        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            return this.Forward(x);
        }

        //Functionとして呼び出された時にバトンを渡す
        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            return this.Backward(gy);
        }

        //Forward
        public override NdArray[] Forward(NdArray[] input)
        {
            NdArray[][] inputData = new NdArray[this.Functions.Length + 1][];
            inputData[0] = input;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                inputData[i + 1] = this.Functions[i].Forward(inputData[i]);
            }

            return inputData[this.Functions.Length];
        }

        //Backward
        public override NdArray[] Backward(NdArray[] backwardResult)
        {
            for (int i = this.Functions.Length - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            return backwardResult;
        }


        //Forward
        public override NdArray Forward(NdArray input)
        {
            NdArray[] inputData = new NdArray[this.Functions.Length + 1];
            inputData[0] = input;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                inputData[i + 1] = this.Functions[i].Forward(inputData[i]);
            }

            return inputData[this.Functions.Length];
        }

        //Backward
        public override NdArray Backward(NdArray backwardResult)
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
        public override NdArray[] Predict(NdArray[] forwardResult)
        {
            foreach (Function function in this.Functions)
            {
                forwardResult = function.Predict(forwardResult);
            }

            return forwardResult;
        }

        //予想を実行する[非バッチ]
        public override NdArray Predict(NdArray input)
        {
            foreach (Function function in this.Functions)
            {
                input = function.Predict(input);
            }

            return input;
        }

        //コピーを作成するメソッド
        public FunctionStack Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
