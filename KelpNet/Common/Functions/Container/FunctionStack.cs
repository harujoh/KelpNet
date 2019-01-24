using System;
using System.Collections.Generic;

namespace KelpNet
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "FunctionStack";

        //すべての層がココにFunctionクラスとして保管される
        public Function<T>[] Functions { get; private set; }

        //コンストラクタ
        public FunctionStack(Function<T>[] functions, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = functions;
        }

        public FunctionStack(Function<T> function, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = new[] { function };
        }

        public FunctionStack(params Function<T>[] functions) : base(FUNCTION_NAME)
        {
            this.Functions = new Function<T>[]{};
            this.Add(functions);
        }

        //頻繁に使用することを想定していないため効率の悪い実装になっている
        public void Add(params Function<T>[] function)
        {
            if (function != null && function.Length > 0)
            {
                List<Function<T>> functionList = new List<Function<T>>();

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

        public void Compress()
        {
            List<Function<T>> functionList = new List<Function<T>>(Functions);

            //層を圧縮
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is CompressibleFunction<T>)
                {
                    if (functionList[i + 1] is CompressibleActivation<T>)
                    {
                        ((CompressibleFunction<T>)functionList[i]).SetActivation((CompressibleActivation<T>)functionList[i + 1]);
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            this.Functions = functionList.ToArray();
        }

        //Forward
        public override NdArray<T>[] Forward(params NdArray<T>[] xs)
        {
            NdArray<T>[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Forward(ys);
            }

            return ys;
        }

        //Backward
        public override void Backward(params NdArray<T>[] ys)
        {
            NdArray<T>.Backward(ys[0]);
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
            foreach (Function<T> function in this.Functions)
            {
                function.ResetState();
            }
        }

        //予想を実行する
        public override NdArray<T>[] Predict(params NdArray<T>[] xs)
        {
            NdArray<T>[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Predict(ys);
            }

            return ys;
        }

        public override void SetOptimizer(params Optimizer<T>[] optimizers)
        {
            foreach (Function<T> function in this.Functions)
            {
                function.SetOptimizer(optimizers);
            }
        }
    }
}
