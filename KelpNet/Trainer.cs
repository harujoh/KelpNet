using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        //学習処理を行う
        public static double Train(FunctionStack functionStack, Array input, Array teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            NdArray lossResult = lossFunction.Evaluate(functionStack.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);

            //Backwardのバッチを実行
            functionStack.Backward(lossResult);

            //更新
            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        //バッチで学習処理を行う
        public static double BatchTrain(FunctionStack functionStack, Array[] input, Array[] teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            NdArray[] lossResult = lossFunction.Evaluate(functionStack.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);

            //Backwardのバッチを実行
            functionStack.Backward(lossResult);

            //更新
            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        //予想を実行する[非バッチ]（外部からの使用を想定してArrayが引数
        public static NdArray Predict(FunctionStack functionStack, Array input)
        {
            return functionStack.Predict(NdArray.FromArray(input));
        }

        //予想を実行する（外部からの使用を想定してArrayが引数
        public static NdArray[] BatchPredict(FunctionStack functionStack, Array[] input)
        {
            NdArray[] ndArrays = new NdArray[input.Length];

            for (int i = 0; i < ndArrays.Length; i++)
            {
                ndArrays[i] = NdArray.FromArray(input[i]);
            }

            return functionStack.Predict(ndArrays);
        }

        //精度測定
        public static double Accuracy(FunctionStack functionStack, Array[] x, int[][] y)
        {
            int matchCount = 0;

            NdArray[] forwardResult = functionStack.Predict(NdArray.FromArray(x));

            for (int i = 0; i < x.Length; i++)
            {
                if (Array.IndexOf(forwardResult[i].Data, forwardResult[i].Data.Max()) == y[i][0])
                {
                    matchCount++;
                }
            }

            return matchCount / (double)x.Length;
        }
    }
}
