using System;
using System.Collections.Generic;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions.Container
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack : Function
    {
        const string FUNCTION_NAME = "FunctionStack";

        //すべての層がココにFunctionクラスとして保管される
        public Function[] Functions { get; private set; }

        //コンストラクタ
        public FunctionStack(Function[] functions, string name = FUNCTION_NAME) : base(name)
        {
            this.Functions = functions;
        }

        public FunctionStack(params Function[] functions) : base(FUNCTION_NAME)
        {
            this.Add(functions);
        }

        //頻繁に使用することを想定していないため効率の悪い実装になっている
        public void Add(params Function[] function)
        {
            if (function != null)
            {
                List<Function> functionList = new List<Function>();

                if (this.Functions != null)
                {
                    functionList.AddRange(this.Functions);
                }

                for (int i = 0; i < function.Length; i++)
                {
                    if (function[i] != null) functionList.Add(function[i]);
                }

                this.Functions = functionList.ToArray();
            }
        }

        public void Compress()
        {
            List<Function> functionList = new List<Function>(Functions);

            //層を圧縮
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is CompressibleFunction)
                {
                    if (functionList[i + 1] is CompressibleActivation)
                    {
                        ((CompressibleFunction)functionList[i]).SetActivation((CompressibleActivation)functionList[i + 1]);
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            this.Functions = functionList.ToArray();
        }

        //Forward
        public override NdArray[] Forward(params NdArray[] xs)
        {
            NdArray[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Forward(ys);
            }

            return ys;
        }

        //Backward
        public override void Backward(params NdArray[] ys)
        {
            NdArray.Backward(ys[0]);
        }

        //重みの更新処理
        public override void Update()
        {
            foreach (var function in Functions)
            {
                function.Update();
            }
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
        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Predict(ys);
            }

            return ys;
        }

        public override void SetOptimizer(params Optimizer[] optimizers)
        {
            foreach (Function function in this.Functions)
            {
                function.SetOptimizer(optimizers);
            }
        }
    }
}
