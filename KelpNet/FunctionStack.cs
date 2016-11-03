using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using KelpNet.Common;

namespace KelpNet
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack : Function
    {
        //すべての層がココにFunctionクラスとして保管される
        public readonly List<Function> Functions = new List<Function>();

        //Optimizerをココで保持する
        private Optimizer[] Optimizers;

        //コンストラクタ
        public FunctionStack(params Function[] functions) : base("FunctionStack")
        {
            //入力された関数を振り分ける
            foreach (Function function in functions)
            {
                //全関数リストへ追加
                this.Functions.Add(function);

                //パラメーターを保持
                Parameters.AddRange(function.Parameters);
            }
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
            foreach (Function function in this.Functions)
            {
                input = function.Forward(input);
            }

            return input;
        }

        //Backward
        public override NdArray[] Backward(NdArray[] backwardResult)
        {
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            return backwardResult;
        }


        //Forward
        public override NdArray Forward(NdArray input)
        {
            foreach (Function function in this.Functions)
            {
                input = function.Forward(input);
            }

            return input;
        }

        //Backward
        public override NdArray Backward(NdArray backwardResult)
        {
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            return backwardResult;
        }

        //Optimizerを設定
        public void SetOptimizer(params Optimizer[] optimizers)
        {
            this.Optimizers = optimizers;
            foreach (var optimizer in optimizers)
            {
                optimizer.SetParameters(Parameters);
            }
        }

        //重みの更新処理
        public void Update()
        {
            //更新実行前に訓練カウントを使って各Functionの傾きを補正
            foreach (var function in this.Functions)
            {
                foreach (OptimizeParameter functionParameter in function.Parameters)
                {
                    for (int k = 0; k < functionParameter.Length; k++)
                    {
                        functionParameter.Grad.Data[k] /= functionParameter.TrainCount;
                    }
                }
            }

            //Optimizerの更新を実行
            foreach (var optimizer in this.Optimizers)
            {
                optimizer.Update();
            }

            //傾きとカウンタをリセット
            this.ClearGrads();

            //ガベージコレクタを明示的に起動
            GC.Collect();
        }

        //傾きの初期化
        public void ClearGrads()
        {
            foreach (var function in this.Functions)
            {
                foreach (OptimizeParameter parameter in function.Parameters)
                {
                    parameter.ClearGrad();
                }
            }
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public override void ResetState()
        {
            foreach (var function in this.Functions)
            {
                function.ResetState();
            }
        }

        //予想を実行する
        public override NdArray[] Predict(NdArray[] forwardResult)
        {
            foreach (Function predictableFunction in this.Functions)
            {
                forwardResult = predictableFunction.Predict(forwardResult);
            }

            return forwardResult;
        }

        //予想を実行する[非バッチ]
        public override NdArray Predict(NdArray input)
        {
            foreach (Function predictableFunction in this.Functions)
            {
                input = predictableFunction.Predict(input);
            }

            return input;
        }

        public void Save(string fileName)
        {
            BinaryFormatter bf = new BinaryFormatter();

            using (Stream stream = File.OpenWrite(fileName))
            {
                bf.Serialize(stream, this);
            }            
        }

        public static FunctionStack Load(string fileName)
        {
            BinaryFormatter bf = new BinaryFormatter();
            FunctionStack result;

            using (Stream stream = File.OpenRead(fileName))
            {
                result = (FunctionStack)bf.Deserialize(stream);
            }

            return result;
        }

        //コピーを作成するメソッド
        public FunctionStack Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
