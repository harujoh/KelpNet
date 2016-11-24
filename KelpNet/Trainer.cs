using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        //ロス関数のデリゲート宣言
        public delegate NdArray[] LossFunction(NdArray[] input, NdArray[] teachSignal, out double loss);
        public delegate NdArray SingleLossFunction(NdArray input, NdArray teachSignal, out double loss);

        public static NdArray[] Forward(Function function, Array[] input)
        {
            return function.Forward(NdArray.FromArray(input));
        }

        public static NdArray Forward(Function function, Array input)
        {
            return function.Forward(NdArray.FromArray(input));
        }

        public static NdArray[] Forward(Function function, Array[] input, Array[] teach, LossFunction lossFunction, out double sumLoss)
        {
            return lossFunction(function.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);
        }

        public static NdArray Forward(Function function, Array input, Array teach, SingleLossFunction lossFunction, out double sumLoss)
        {
            return lossFunction(function.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);
        }

        public static NdArray[] Backward(Function function, Array[] input)
        {
            return function.Backward(NdArray.FromArray(input));
        }

        public static NdArray Backward(Function function, Array input)
        {
            return function.Backward(NdArray.FromArray(input));
        }

        //バッチで学習処理を行う
        public static double Train(FunctionStack functionStack, Array input, Array teach, SingleLossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            NdArray lossResult = lossFunction(functionStack.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);

            //Backwardのバッチを実行
            functionStack.Backward(lossResult);

            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        public static double Train(FunctionStack functionStack, Array[] input, Array[] teach, LossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            NdArray[] lossResult = lossFunction(functionStack.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);

            //Backwardのバッチを実行
            functionStack.Backward(lossResult);

            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        //非バッチで学習処理を行う
        public static double Train(FunctionStack functionStack, NdArray input, NdArray teach, SingleLossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardを実行
            NdArray lossResult = lossFunction(functionStack.Forward(input), teach, out sumLoss);

            //Backwardを実行
            functionStack.Backward(lossResult);

            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        //精度測定
        public static double Accuracy(FunctionStack functionStack, Array[] x, int[][] y)
        {
            return Accuracy(functionStack, NdArray.FromArray(x), y);
        }

        public static double Accuracy(FunctionStack functionStack, NdArray[] x, int[][] y)
        {
            int matchCount = 0;

            NdArray[] forwardResult = functionStack.Predict(x);

            for (int i = 0; i < x.Length; i++)
            {
                if (Array.IndexOf(forwardResult[i].Data, forwardResult[i].Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            }

            return matchCount / (double)x.Length;
        }

        //予想を実行する（外部からの使用を想定してArrayが引数
        public static NdArray[] Predict(FunctionStack functionStack, Array[] input)
        {
            NdArray[] ndArrays = new NdArray[input.Length];

            for (int i = 0; i < ndArrays.Length; i++)
            {
                ndArrays[i] = NdArray.FromArray(input[i]);
            }

            return functionStack.Predict(ndArrays);
        }

        //予想を実行する[非バッチ]（外部からの使用を想定してArrayが引数
        public static NdArray Predict(FunctionStack functionStack, Array input)
        {
            return functionStack.Predict(NdArray.FromArray(input));
        }

    }
}
