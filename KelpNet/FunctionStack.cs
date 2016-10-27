using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using KelpNet.Common;

namespace KelpNet
{
    //層を積み上げるこのライブラリのメインとなるクラス
    [Serializable]
    public class FunctionStack
    {
        //ロス関数のデリゲート宣言
        public delegate NdArray[] LossFunction(NdArray[] input, NdArray[] teachSignal, out double loss);
        public delegate NdArray SingleLossFunction(NdArray input, NdArray teachSignal, out double loss);

        //すべての層がココにFunctionクラスとして保管される
        public readonly List<Function> Functions = new List<Function>();

        //Optimizerをココで保持する
        private Optimizer[] _optimizers;

        //更新対象となるパラメータを保持
        public List<OptimizeParameter> Parameters = new List<OptimizeParameter>();

        //コンストラクタ
        public FunctionStack(params Function[] functions)
        {
            //入力された関数を振り分ける
            foreach (Function function in functions)
            {
                //全関数リストへ追加
                this.Functions.Add(function);

                //パラメーターを保持
                this.Parameters.AddRange(function.Parameters);
            }
        }

        //Optimizerを設定
        public void SetOptimizer(params Optimizer[] optimizers)
        {
            this._optimizers = optimizers;
            foreach (var optimizer in optimizers)
            {
                optimizer.SetParameters(this.Parameters);
            }
        }

        //Forward
        public NdArray[] Forward(Array[] input, Array[] teach, LossFunction lossFunction, out double sumLoss)
        {
            NdArray[] inputData = new NdArray[input.Length];
            for (int i = 0; i < inputData.Length; i++)
            {
                inputData[i] = NdArray.FromArray(input[i]);
            }

            foreach (Function function in this.Functions)
            {
                inputData = function.Forward(inputData);
            }

            NdArray[] teachArray = new NdArray[teach.Length];
            for (int i = 0; i < teach.Length; i++)
            {
                teachArray[i] = NdArray.FromArray(teach[i]);
            }

            //デリゲートで入力されたロス関数を実行
            return lossFunction(inputData, teachArray, out sumLoss);
        }

        //Forward
        public NdArray Forward(Array input, Array teach, SingleLossFunction lossFunction, out double sumLoss)
        {
            NdArray inputData = NdArray.FromArray(input);

            foreach (Function function in this.Functions)
            {
                inputData = function.Forward(inputData);
            }

            var teachArray = NdArray.FromArray(teach);

            //デリゲートで入力されたロス関数を実行
            return lossFunction(inputData, teachArray, out sumLoss);
        }

        //Backward
        public void Backward(NdArray[] backwardResult)
        {
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }
        }

        //Backward
        public void Backward(NdArray backwardResult)
        {
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }
        }

        //バッチで学習処理を行う
        public double Train(Array[] input, Array[] teach, LossFunction lossFunction)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            var backwardResult = this.Forward(input, teach, lossFunction, out sumLoss);

            //Backwardのバッチを実行
            this.Backward(backwardResult);

            return sumLoss;
        }

        //非バッチで学習処理を行う
        public double Train(Array input, Array teach, SingleLossFunction lossFunction)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardを実行
            var backwardResult = this.Forward(input, teach, lossFunction, out sumLoss);

            //Backwardを実行
            this.Backward(backwardResult);

            return sumLoss;
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
            foreach (var optimizer in this._optimizers)
            {
                optimizer.Update();
            }

            //傾きをリセット
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
        public void ResetState()
        {
            foreach (var function in this.Functions)
            {
                function.ResetState();
            }
        }

        //予想を実行する（外部からの使用を想定してArrayが引数
        public NdArray[] Predict(Array[] input)
        {
            NdArray[] ndArrays = new NdArray[input.Length];
            for (int i = 0; i < ndArrays.Length; i++)
            {
                ndArrays[i] = NdArray.FromArray(input[i]);
            }

            return this.Predict(ndArrays);
        }

        //予想を実行する[非バッチ]（外部からの使用を想定してArrayが引数
        public NdArray Predict(Array input)
        {
            return this.Predict(NdArray.FromArray(input));
        }

        //予想を実行する
        public NdArray[] Predict(NdArray[] input)
        {
            NdArray[] forwardResult = new NdArray[input.Length];
            for (int i = 0; i < forwardResult.Length;i++)
            {
                forwardResult[i] = new NdArray(input[i]);
            }

            foreach (Function predictableFunction in this.Functions)
            {
                forwardResult = predictableFunction.Predict(forwardResult);
            }

            return forwardResult;
        }

        //予想を実行する[非バッチ]
        public NdArray Predict(NdArray input)
        {
            NdArray forwardResult = new NdArray(input);

            foreach (Function predictableFunction in this.Functions)
            {
                forwardResult = predictableFunction.Predict(forwardResult);
            }

            return forwardResult;
        }

        //精度測定
        public double Accuracy(Array[] x, int[][] y)
        {
            int matchCount = 0;

            var forwardResult = this.Predict(x);

            for (int i = 0; i < x.Length; i++)
            {
                if (Array.IndexOf(forwardResult[i].Data, forwardResult[i].Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            }

            return matchCount / (double)x.Length;
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
