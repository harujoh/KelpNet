using System;
using System.Collections.Generic;

namespace KelpNet.CPU
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack : Function
    {
        protected const string FUNCTION_NAME = "FunctionStack";

        //すべての層がココにFunctionクラスとして保管される
        public Function[] Functions { get; set; }

        //コンストラクタ
        public FunctionStack(Function[] functions, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = functions;
        }

        public FunctionStack(Function function, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = new[] { function };
        }

        public FunctionStack(params Function[] functions) : base(FUNCTION_NAME)
        {
            this.Functions = new Function[] { };
            this.Add(functions);
        }

        //頻繁に使用することを想定していないため効率の悪い実装になっている
        public void Add(params Function[] function)
        {
            if (function != null && function.Length > 0)
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

                InputNames = Functions[0].InputNames;
                OutputNames = Functions[Functions.Length - 1].OutputNames;
            }
        }

        public virtual void Compress()
        {
            List<Function> functionList = new List<Function>(Functions);

            //層を圧縮
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is ICompressibleFunction compressibleFunction)
                {
                    if (compressibleFunction.Activation == null && functionList[i + 1] is ICompressibleActivation compressibleActivation)
                    {
                        compressibleFunction.Activation = compressibleActivation;
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

            ResetState();
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
