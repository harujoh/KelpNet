using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KelpNet.Interface;

namespace KelpNet
{
    //層を積み上げるこのライブラリのメインとなるクラス
    public class FunctionStack
    {
        //ロス関数のデリゲート宣言
        public delegate NdArray LossFunction(NdArray input, NdArray teachSignal, out double loss);

        //バッチ実行用に保管
        private readonly List<FunctionPare> _functionPares = new List<FunctionPare>();

        //すべての層がココにFunctionクラスとして保管される
        public readonly List<Function> Functions = new List<Function>();

        //学習用の関数を除く関数がココにPredictableFunctionとして保管される（現在はDropoutを実行しないために用意）
        private readonly List<IPredictableFunction> _predictableFunctions = new List<IPredictableFunction>();

        //Optimizerをココで保持する
        private Optimizer _optimizer;

        //更新対象となるパラメータを保持
        public List<OptimizeParameter> Parameters = new List<OptimizeParameter>();

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
                this._functionPares.Add(new FunctionPare());
            }

            //入力された関数を振り分ける
            foreach (Function function in functions)
            {
                //逐次実行タスク用
                this.StackSingleTaskFunction(function);

                //バッチ実行タスク用
                this.StackBatchTaskFunction(function, isPreFuncBatch);

                //パラメーターを保持
                this.Parameters.AddRange(function.Parameters);

                //フラグを設定
                isPreFuncBatch = batchFunction != null;
            }
        }

        //コピーを作成するメソッド
        public FunctionStack Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }

        //Optimizerを設定
        public void SetOptimizer(Optimizer optimizer)
        {
            this._optimizer = optimizer;
            this._optimizer.SetParameters(this.Parameters);
        }

        //シングルタスク用の層を積み上げる
        public void StackSingleTaskFunction(Function function)
        {
            //全関数リストへ追加
            this.Functions.Add(function);

            //予測処理実行用のリストへ追加
            var predictableFunction = function as IPredictableFunction;
            if (predictableFunction != null)
            {
                this._predictableFunctions.Add(predictableFunction);
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
                    this._functionPares.Add(new FunctionPare());
                }

                //ペアにバッチ関数を追加
                this._functionPares[this._functionPares.Count - 1].BatchFunctions.Add(batchFunction);
            }
            else
            {
                //ペアに関数を追加
                this._functionPares[this._functionPares.Count - 1].SoloFunctions.Add(function);
            }
        }

        //傾きの初期化
        public void ClearGrads()
        {
            foreach (var function in this.Functions)
            {
                for (int j = 0; j < function.Parameters.Count; j++)
                {
                    function.Parameters[j].ClearGrad();
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
        public NdArray Predict(Array input, int batchID = 0)
        {
            return this.Predict(NdArray.FromArray(input), batchID);
        }

        //予想を実行する
        public NdArray Predict(NdArray input, int batchID = 0)
        {
            NdArray forwardResult = input;

            foreach (IPredictableFunction predictableFunction in this._predictableFunctions)
            {
                forwardResult = predictableFunction.Predict(forwardResult, batchID);
            }

            return forwardResult;
        }

        //訓練を実施する　引数のデリゲートにロス関数を入力する
        //public delegate NdArray LossFunction(NdArray input, NdArray teachSignal, out double loss);
        public double Train(Array input, Array teach, LossFunction lossFunction)
        {
            //全層の『入力』と『出力』を全て保存するため＋１
            NdArray[] inputData = new NdArray[this.Functions.Count + 1];

            //入力値を保存
            inputData[0] = NdArray.FromArray(input);

            //forwardを実行
            for (int i = 0; i < this.Functions.Count; i++)
            {
                //出力を次層の入力として保存する
                inputData[i + 1] = this.Functions[i].Forward(inputData[i]);
            }

            //戻り値の誤差用
            double loss;

            //デリゲートで入力されたロス関数を実行
            NdArray backwardResult = lossFunction(inputData[this.Functions.Count], NdArray.FromArray(teach), out loss);

            //backwardを実行
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }

            return loss;
        }

        public void InitBatch(int batchSize)
        {
            //入出力を初期化
            foreach (Function function in this.Functions)
            {
                function.InitBatch(batchSize);
            }
        }

        //Forwardのバッチ版
        public NdArray[] BatchForward(Array[] input, Array[] teach, LossFunction lossFunction, out List<double> sumLoss)
        {
            int batchCount = input.Length;

            //全層の『入力』と『出力』を全て保存するため＋１
            NdArray[][] inputData = new NdArray[this.Functions.Count + 1][];
            for (int i = 0; i < this.Functions.Count + 1; i++)
            {
                inputData[i] = new NdArray[batchCount];
            }

            //最初のデータを保存
            for (int i = 0; i < batchCount; i++)
            {
                inputData[0][i] = NdArray.FromArray(input[i]);
            }

            int functionCount = 0;

            //forwardを実行
            foreach (FunctionPare functionPare in this._functionPares)
            {
                //まずバッチ専用関数を処理
                foreach (IBatchFunction batchFunction in functionPare.BatchFunctions)
                {
                    inputData[functionCount + 1] = batchFunction.BatchForward(inputData[functionCount]);

                    functionCount++;
                }

                //その後、その他の関数を処理
#if DEBUG
                for (int k = 0; k < batchCount; k++)
                {
                    for (int j = 0; j < functionPare.SoloFunctions.Count; j++)
                    {
                        inputData[functionCount + j + 1][k] = functionPare.SoloFunctions[j].Forward(inputData[functionCount + j][k], k);
                    }
                }
#else
                Parallel.For(0, batchCount, k =>
                {
                    for (int j = 0; j < functionPare.SoloFunctions.Count; j++)
                    {
                        inputData[functionCount + j + 1][k] = functionPare.SoloFunctions[j].Forward(inputData[functionCount + j][k], k);
                    }
                });
#endif
                functionCount += functionPare.SoloFunctions.Count;
            }

            //戻り値の誤差用
            sumLoss = new List<double>();

            NdArray[] backwardResult = new NdArray[batchCount];
            for (int i = 0; i < backwardResult.Length; i++)
            {
                double loss;
                //デリゲートで入力されたロス関数を実行
                backwardResult[i] = lossFunction(inputData[functionCount][i], NdArray.FromArray(teach[i]), out loss);

                sumLoss.Add(loss);
            }

            return backwardResult;
        }

        //Backwardのバッチ版
        public void BatchBackward(NdArray[] backwardResult)
        {
            //backwardを実行
            for (int i = this._functionPares.Count - 1; i >= 0; i--)
            {
                //バックワードは逆にその他の関数から処理
#if DEBUG
                for (int k = 0; k < backwardResult.Length; k++)
                {
                    for (int j = this._functionPares[i].SoloFunctions.Count - 1; j >= 0; j--)
                    {
                        backwardResult[k] = this._functionPares[i].SoloFunctions[j].Backward(backwardResult[k], k);
                    }
                }
#else
                Parallel.For(0, backwardResult.Length, k =>
                {
                    for (int j = this._functionPares[i].SoloFunctions.Count - 1; j >= 0; j--)
                    {
                        backwardResult[k] = this._functionPares[i].SoloFunctions[j].Backward(backwardResult[k], k);
                    }
                });
#endif

                //その後、バッチ関数を処理
                for (int j = this._functionPares[i].BatchFunctions.Count - 1; j >= 0; j--)
                {
                    backwardResult = this._functionPares[i].BatchFunctions[j].BatchBackward(backwardResult);
                }
            }
        }

        //バッチで学習処理を行う
        public List<double> BatchTrain(Array[] input, Array[] teach, LossFunction lossFunction)
        {
            //結果の誤差保存用
            List<double> sumLoss;

            //入出力を初期化
            foreach (Function function in this.Functions)
            {
                function.InitBatch(input.Length);
            }

            //Forwardのバッチを実行
            var backwardResult = this.BatchForward(input, teach, lossFunction, out sumLoss);

            //Backwardのバッチを実行
            this.BatchBackward(backwardResult);

            return sumLoss;
        }

        //重みの更新処理
        public void Update()
        {
            //更新実行前にバッチカウントを使って各Functionの傾きを補正
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

            //宣言されているOptimizerの更新を実行
            this._optimizer.Update();

            //傾きをリセット
            this.ClearGrads();
        }

        //精度測定
        public double Accuracy(Array[] x, byte[][] y)
        {
            int matchCount = 0;

            //入出力を初期化
            foreach (Function function in this.Functions)
            {
                function.InitBatch(x.Length);
            }

#if DEBUG
            for (int i = 0; i < x.Length; i++)
            {
                var forwardResult = this.Predict(x[i], i);

                if (Array.IndexOf(forwardResult.Data, forwardResult.Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            }
#else
            Parallel.For(0, x.Length, i =>
            {
                var forwardResult = this.Predict(x[i], i);

                if (Array.IndexOf(forwardResult.Data, forwardResult.Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            });
#endif

            return matchCount / (double)x.Length;
        }
    }
}
