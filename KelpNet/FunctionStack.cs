using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KelpNet.Optimizers;

namespace KelpNet
{
    //層を積み上げるこのライブラリのメインとなるクラス
    public class FunctionStack
    {
        //ロス関数のデリゲート宣言
        public delegate NdArray LossFunction(NdArray input, NdArray teachSignal, out double loss);

        //ファンクションペア（非バッチ処理をまとめ、極力複数層に渡った並列処理を実現する
        public class FunctionPare
        {
            public List<IBatchFunction> BatchFunctions = new List<IBatchFunction>();
            public List<Function> SoloFunctions = new List<Function>();
        }

        //バッチ実行用に保管
        public readonly List<FunctionPare> FunctionPares = new List<FunctionPare>();

        //すべての層がココにFunctionクラスとして保管される
        public readonly List<Function> Functions = new List<Function>();

        //学習用の関数を除く関数がココにPredictableFunctionとして保管される（現在はDropoutを実行しないために用意）
        public readonly List<PredictableFunction> PredictableFunctions = new List<PredictableFunction>();

        //Updateを行わずに実行されたTrainの回数をカウントし、バッチ更新時に使用する
        private int _batchCount = 0;

        //Optimizerをココで保持する。デフォルトはSGD
        private Optimizer _optimizer = new SGD();

        //コンストラクタ
        public FunctionStack(params Function[] functions)
        {
            //バッチファンクションが連続した時に無駄にペアを増やさないためのフラグ
            bool isPreFuncBatch = false;

            //最初の層はバッチ関数か
            var batchFunction = functions[0] as IBatchFunction;
            if (batchFunction == null)
            {
                //最初が通常の層だったら事前にペアを追加
                this.FunctionPares.Add(new FunctionPare());
            }

            //入力された関数を振り分ける
            foreach (Function function in functions)
            {
                //シングルタスク用
                this.StackSingleTaskFunction(function);

                //バッチタスク用
                this.StackBatchTaskFunction(function, isPreFuncBatch);

                //フラグを設定
                isPreFuncBatch = batchFunction != null;
            }
        }

        //Optimizerを設定
        public void SetOptimizer(Optimizer optimizer)
        {
            this._optimizer = optimizer;
            this._optimizer.Initialize(this);
        }

        //シングルタスク用の層を積み上げる
        public void StackSingleTaskFunction(Function function)
        {
            //全関数リストへ追加
            this.Functions.Add(function);

            //予測処理実行用のリストへ追加
            var predictableFunction = function as PredictableFunction;
            if (predictableFunction != null)
            {
                this.PredictableFunctions.Add(predictableFunction);
            }
        }

        //バッチタスク用の層を積み上げる
        public void StackBatchTaskFunction(Function function, bool preFuncIsBatch)
        {
            //バッチ関数が入ってきたらペアを追加
            var batchFunction = function as IBatchFunction;
            if (batchFunction != null)
            {
                //バッチ関数が連続していないかチェック
                if (!preFuncIsBatch)
                {
                    this.FunctionPares.Add(new FunctionPare());
                }

                //ペアにバッチ関数を追加
                this.FunctionPares[this.FunctionPares.Count - 1].BatchFunctions.Add(batchFunction);
            }
            else
            {
                //ペアに関数を追加
                this.FunctionPares[this.FunctionPares.Count - 1].SoloFunctions.Add(function);
            }
        }

        //傾きの初期化
        public void ClearGrads()
        {
            //バッチカウントもリセット
            this._batchCount = 0;

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
            return this.Predict(NdArray.FromArray(input));
        }

        //予想を実行する
        public NdArray Predict(NdArray input)
        {
            NdArray forwardResult = input;

            foreach (PredictableFunction predictableFunction in this.PredictableFunctions)
            {
                forwardResult = predictableFunction.Predict(forwardResult);
            }

            return forwardResult;
        }

        //訓練を実施する　引数のデリゲートにロス関数を入力する
        //public delegate NdArray LossFunction(NdArray input, NdArray teachSignal, out double loss);
        public double Train(Array input, Array teach, LossFunction lossFunction)
        {
            //全層の『入力』と『出力』を全て保存するため＋１
            NdArray[] InputData = new NdArray[this.Functions.Count + 1];

            //入力値を保存
            InputData[0] = NdArray.FromArray(input);

            //forwardを実行
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
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            //実行回数をカウント
            this._batchCount++;

            return loss;
        }

        //バッチで学習処理を行う
        public List<double> BatchTrain(Array[] input, Array[] teach, LossFunction lossFunction)
        {
            int batchCount = input.Length;

            //入出力を初期化
            foreach (Function function in this.Functions)
            {
                function.InitBatch(batchCount);
            }

            //全層の『入力』と『出力』を全て保存するため＋１
            NdArray[][] InputData = new NdArray[this.Functions.Count + 1][];
            for (int i = 0; i < this.Functions.Count + 1; i++)
            {
                InputData[i] = new NdArray[batchCount];
            }

            for (int i = 0; i < batchCount; i++)
            {
                InputData[0][i] = NdArray.FromArray(input[i]);
            }

            int functionCount = 0;

            //forwardを実行
            foreach (FunctionPare functionPare in this.FunctionPares)
            {
                foreach (IBatchFunction batchFunction in functionPare.BatchFunctions)
                {
                    InputData[functionCount + 1] = batchFunction.BatchForward(InputData[functionCount]);

                    functionCount++;
                }

                Parallel.For(0, batchCount, k =>
                {
                    for (int j = 0; j < functionPare.SoloFunctions.Count; j++)
                    {
                        InputData[functionCount + j + 1][k] = functionPare.SoloFunctions[j].Forward(InputData[functionCount + j][k], k);
                    }
                });

                functionCount += functionPare.SoloFunctions.Count;
            }

            //戻り値の誤差用
            List<double> sumLoss = new List<double>();

            NdArray[] backwardResult = new NdArray[batchCount];
            for (int i = 0; i < backwardResult.Length; i++)
            {
                double loss;
                //デリゲートで入力されたロス関数を実行
                backwardResult[i] = lossFunction(InputData[functionCount][i], NdArray.FromArray(teach[i]), out loss);

                sumLoss.Add(loss);
            }

            //backwardを実行
            for (int i = this.FunctionPares.Count - 1; i >= 0; i--)
            {
                Parallel.For(0, batchCount, k =>
                {
                    for (int j = this.FunctionPares[i].SoloFunctions.Count - 1; j >= 0; j--)
                    {
                        backwardResult[k] = this.FunctionPares[i].SoloFunctions[j].Backward(backwardResult[k], k);
                    }
                });

                functionCount -= this.FunctionPares[i].SoloFunctions.Count;

                for (int j = this.FunctionPares[i].BatchFunctions.Count - 1; j >= 0; j--)
                {
                    backwardResult = this.FunctionPares[i].BatchFunctions[j].BatchBackward(backwardResult, InputData[i], InputData[i + 1]);
                    functionCount--;
                }
            }

            //実行回数をカウント
            this._batchCount += batchCount;

            return sumLoss;
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
                        functionParameter.Grad.Data[k] /= this._batchCount;
                    }
                }
            }

            //宣言されているOptimizerの更新を実行
            this._optimizer.Update(this.Functions);

            //傾きをリセット
            this.ClearGrads();
        }

        //精度測定
        public double Accuracy(Array[] x, byte[][] y)
        {
            int matchCount = 0;

            Parallel.For(0, x.Length, i =>
            {
                var forwardResult = this.Predict(x[i]);

                if (Array.IndexOf(forwardResult.Data, forwardResult.Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            });

            return matchCount / (double)x.Length;
        }
    }
}
