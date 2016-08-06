using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KelpNet.Functions;
using KelpNet.Optimizers;

namespace KelpNet
{
    //層を積み上げるこのライブラリのメインとなるクラス
    public class FunctionStack
    {
        //すべての層がココにFunctionクラスとして保管される
        public readonly List<Function> Functions = new List<Function>();

        //学習用の関数を除く関数がココにPredictableFunctionとして保管される（現在はDropoutを実行しないために用意）
        public readonly List<IPredictableFunction> PredictableFunctions = new List<IPredictableFunction>();

        //Updateを行わずに実行されたTrainの回数をカウントし、バッチ更新時に使用する
        private int BatchCount = 0;

        //Optimizerをココで保持する。デフォルトはSGD
        private Optimizer optimizer = new SGD();

        public void SetOptimizer(Optimizer optimizer)
        {
            this.optimizer = optimizer;
            this.optimizer.Initialize(this);
        }

        //コンストラクタ
        public FunctionStack(params Function[] functions)
        {
            //入力された関数を振り分ける
            foreach (Function function in functions)
            {
                this.StackFunction(function);
            }
        }

        //層を積み上げる
        public void StackFunction(Function function)
        {
            //全関数リストへ追加
            this.Functions.Add(function);

            //予測処理実行用のリストへ追加
            var PredictableFunction = function as IPredictableFunction;
            if (PredictableFunction != null)
            {
                this.PredictableFunctions.Add(PredictableFunction);
            }
        }

        //傾きの初期化
        public void ZeroGrads()
        {
            //バッチカウントもリセット
            this.BatchCount = 0;

            foreach (var function in this.Functions)
            {                
                for (int j = 0; j < function.Parameters.Count; j++)
                {
                    function.Parameters[j].Grad.Fill(0);
                }
            }
        }

        //予想を実行する（外部からの使用を想定してArrayが引数
        public NdArray Predict(Array input)
        {
            return Predict(NdArray.FromArray(input));
        }

        //予想を実行する
        public NdArray Predict(NdArray input)
        {
            NdArray forwardResult = input;

            foreach (IPredictableFunction predictableFunction in this.PredictableFunctions)
            {
                forwardResult = predictableFunction.Predict(forwardResult);
            }

            return forwardResult;
        }

        //訓練を実施する　引数のデリゲートにロス関数を入力する
        public delegate NdArray LossFunction(NdArray input, NdArray teachSignal, out double loss);
        public double Train(Array input, Array teach, LossFunction lossFunction)
        {
            //全層の『入力』と『出力』を全て保存するため＋１
            NdArray[] InputData = new NdArray[this.Functions.Count + 1];

            //forwardを実行
            InputData[0] = NdArray.FromArray(input);

            for (int i = 0; i < this.Functions.Count; i++)
            {
                //出力を次層の入力として保存する
                InputData[i + 1] = this.Functions[i].Forward(InputData[i]);
            }

            //戻り値の誤差用
            double loss;

            //デリゲートで入力されたロス関数を実行
            NdArray backwardResult = lossFunction(InputData[this.Functions.Count], NdArray.FromArray(teach), out loss);

            //backwardを実行
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult, InputData[i], InputData[i + 1]);
            }

            //実行回数をカウント
            this.BatchCount++;

            return loss;
        }
        
        //並列処理で早くなりそうな名前だが、並列実行が層単位となるため、遅い
        public double BatchTrain(Array[] input, Array[] teach, LossFunction lossFunction, int batchCount = -1, int startOffset = 0, bool shuffle = false)
        {
            //todo 範囲チェック
            if (batchCount == -1)
            {
                batchCount = input.Length;
            }

            //全層の『入力』と『出力』を全て保存するため＋１
            NdArray[][] InputData = new NdArray[this.Functions.Count + 1][];
            NdArray[] backwardResult = new NdArray[batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                InputData[0][i] = NdArray.FromArray(input[startOffset + i]);
            }

            for (int j = 0; j < this.Functions.Count; j++)
            {
                InputData[j + 1] = this.Functions[j].BatchForward(InputData[j]);
            }

            //戻り値の誤差用
            double sumLoss = 0;

            //for (int i = 0; i < batchCount; i++)
            Parallel.For(0, backwardResult.Length, i =>
            {
                double loss;
                //デリゲートで入力されたロス関数を実行
                backwardResult[i] = lossFunction(InputData[i][this.Functions.Count], NdArray.FromArray(teach[startOffset + i]), out loss);
                sumLoss += loss;
            });

            //backwardを実行
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].BatchBackward(backwardResult, InputData[i], InputData[i + 1]);
            }

            //実行回数をカウント
            this.BatchCount = batchCount;

            //Updateもまとめて実行
            this.Update();

            return sumLoss / batchCount;
        }

        //重みの更新処理
        public void Update()
        {
            //更新実行前にバッチカウントを使って各Functionの傾きを補正
            foreach (var function in this.Functions)
            {
                foreach (Function.Parameter functionParameter in function.Parameters)
                {
                    for (int k = 0; k < functionParameter.Length; k++)
                    {
                        functionParameter.Grad.Data[k] /= this.BatchCount;
                    }
                }
            }

            //宣言されているOptimizerの更新を実行
            optimizer.Update(this.Functions);

            //傾きをリセット
            this.ZeroGrads();
        }

        //精度測定
        public double Accuracy(Array[] x, byte[][] y)
        {
            int matchCount = 0;

            Parallel.For(0, x.Length, i =>
            {
                var forwardResult = Predict(x[i]);

                if (Array.IndexOf(forwardResult.Data, forwardResult.Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            });

            return matchCount / (double)x.Length;
        }
    }
}
