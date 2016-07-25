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
        
        //重みと傾きを持つものはココにOptimizableFunctionクラスとして保管される
        public readonly List<OptimizableFunction> OptimizableFunctions = new List<OptimizableFunction>();
        
        //学習用の関数を除く関数がココにPredictableFunctionとして保管される（現在はDropoutを実行しないために用意）
        public readonly List<IPredictableFunction> PredictableFunctions = new List<IPredictableFunction>();

        //Updateを行わずに実行されたTrainの回数をカウントし、バッチ更新時に使用する
        private int batchCount = 0;

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

            //学習対象関数のリストへ追加
            var optimizableFunction = function as OptimizableFunction;
            if (optimizableFunction != null)
            {
                this.OptimizableFunctions.Add(optimizableFunction);
            }

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
            batchCount = 0;

            foreach (var function in this.OptimizableFunctions)
            {
                function.gW.Fill(0);

                if (function.b != null)
                {
                    function.gb.Fill(0);
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
            NdArray forwardResult = this.PredictableFunctions[0].Predict(input);
            for (int i = 1; i < this.PredictableFunctions.Count; i++)
            {
                forwardResult = this.PredictableFunctions[i].Predict(forwardResult);
            }

            return forwardResult;
        }

        //訓練を実施する　引数のデリゲートにロス関数を入力する
        public delegate NdArray LossFunction(NdArray input, NdArray teachSignal, out double loss);
        public double Train(Array input, Array teach, LossFunction lossFunction)
        {
            //forwardを実行
            NdArray forwardResult = this.Functions[0].Forward(NdArray.FromArray(input));
            for (int i = 1; i < this.Functions.Count; i++)
            {
                forwardResult = this.Functions[i].Forward(forwardResult);
            }

            //戻り値の誤差用
            double loss;

            //デリゲートで入力されたロス関数を実行
            NdArray backwardResult = lossFunction(forwardResult, NdArray.FromArray(teach), out loss);

            //backwardを実行
            for (int i = this.Functions.Count - 1; i >= 0; i--)
            {
                backwardResult = this.Functions[i].Backward(backwardResult);
            }
            
            //実行回数をカウント
            batchCount++;

            return loss;
        }

        //重みの更新処理
        public void Update()
        {
            //更新実行前にバッチカウントを使って各Functionの傾きを補正
            foreach (OptimizableFunction optimizableFunction in this.OptimizableFunctions)
            {
                for (int j = 0; j < optimizableFunction.gW.Length; j++)
                {
                    optimizableFunction.gW.Data[j] /= this.batchCount;
                }

                if (optimizableFunction.gb != null)
                {
                    for (int j = 0; j < optimizableFunction.gb.Length; j++)
                    {
                        optimizableFunction.gb.Data[j] /= this.batchCount;
                    }
                }
            }

            //宣言されているOptimizerの更新を実行
            optimizer.Update(this.OptimizableFunctions);

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
