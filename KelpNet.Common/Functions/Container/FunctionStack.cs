using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [DataContract(Name = "FunctionStack", Namespace = "KelpNet")]
    public class FunctionStack<T> : Function<T> where T : unmanaged, IComparable<T>
    {
        protected const string FUNCTION_NAME = "FunctionStack";

        //すべての層がココにFunctionクラスとして保管される
        [DataMember]
        public Function<T>[] Functions { get; set; }

        //コンストラクタ
        public FunctionStack(Function<T>[] functions, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = functions;
            InitFunc(new StreamingContext());
            this.SetParam();
        }

        public FunctionStack(Function<T> function, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Functions = new[] { function };
            InitFunc(new StreamingContext());
            this.SetParam();
        }

        public FunctionStack(params Function<T>[] functions) : base(FUNCTION_NAME)
        {
            this.Functions = new Function<T>[] { };
            this.Add(functions);
            this.InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            this.Forward = ForwardFS;
            this.Backward = BackwardFS;
            this.Predict = PredictFS;
        }

        void SetParam()
        {
            int paramCount = 0;
            for (int i = 0; i < Functions.Length; i++)
            {
                paramCount += Functions[i].Parameters.Length;
            }

            Parameters = new NdArray<T>[paramCount];

            paramCount = 0;
            for (int i = 0; i < Functions.Length; i++)
            {
                for (int j = 0; j < Functions[i].Parameters.Length; j++)
                {
                    Parameters[paramCount++] = Functions[i].Parameters[j];
                }
            }
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
                    if (function[i] != null)
                    {
                        functionList.Add(function[i]);
                    }
                }

                this.Functions = functionList.ToArray();

                InputNames = Functions[0].InputNames;
                OutputNames = Functions[Functions.Length - 1].OutputNames;

                this.SetParam();
            }
        }

        public virtual void Compress()
        {
            List<Function<T>> functionList = new List<Function<T>>(Functions);

            //層を圧縮
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is ICompressibleFunction<T> compressibleFunction)
                {
                    if (compressibleFunction.Activation == null && functionList[i + 1] is ICompressibleActivation<T> compressibleActivation)
                    {
                        compressibleFunction.Activation = compressibleActivation;
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            this.Functions = functionList.ToArray();
        }

        //Forward
        public NdArray<T>[] ForwardFS(params NdArray<T>[] xs)
        {
            NdArray<T>[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Forward(ys);
            }

            return ys;
        }

        //Backward
        public void BackwardFS(params NdArray<T>[] ys)
        {
            NdArray.Backward(ys[0]);
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
        public NdArray<T>[] PredictFS(params NdArray<T>[] xs)
        {
            NdArray<T>[] ys = xs;

            for (int i = 0; i < this.Functions.Length; i++)
            {
                ys = this.Functions[i].Predict(ys);
            }

            return ys;
        }
    }
}
