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
        readonly private OptimizeParameter[] optimizer_params;

        public override OptimizeParameter[] GetOptimizeParameters()
        {
            return optimizer_params;
        }
        //すべての層がココにFunctionクラスとして保管される
        public readonly Function[] Functions;

        //コンストラクタ
        public FunctionStack(params Function[] functions) : base("FunctionStack")
        {
            this.Functions = functions;

            //パラメーター参照用の入れ物を作成
            List<OptimizeParameter> parameters = new List<OptimizeParameter>();
            foreach (Function function in functions)
            {
                parameters.AddRange(function.GetOptimizeParameters());
            }
            this.optimizer_params = parameters.ToArray();            
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
            for (int i = this.Functions.Length - 1; i >= 0; i--)
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
            for (int i = this.Functions.Length - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            return backwardResult;
        }

        //重みの更新処理
        public void Update(params Optimizer[] optimizers)
        {
            OptimizeParameter[] Parameters = this.GetOptimizeParameters();
            //更新実行前に訓練カウントを使って各Functionの傾きを補正
            foreach (OptimizeParameter parameter in Parameters)
            {
                for (int j = 0; j < parameter.Length; j++)
                {
                    parameter.Grad.Data[j] /= parameter.TrainCount;
                }
            }

            //Optimizerの更新を実行
            foreach (var optimizer in optimizers)
            {
                optimizer.Update();
            }

            //傾きとカウンタをリセット
            this.ClearGrads();
        }

        //傾きの初期化
        public void ClearGrads()
        {
            OptimizeParameter[] Parameters = this.GetOptimizeParameters();
            foreach (OptimizeParameter parameter in Parameters)
            {
                parameter.ClearGrad();
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
