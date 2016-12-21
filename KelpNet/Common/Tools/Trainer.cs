using System;
using System.Linq;
using KelpNet.Common.Loss;
using KelpNet.Functions;

namespace KelpNet.Common.Tools
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        //学習処理を行う
        public static double Train(FunctionStack functionStack, Array input, Array teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, NdArray.FromArray(input), NdArray.FromArray(teach), lossFunction, isUpdate);
        }

        public static double Train(FunctionStack functionStack, NdArray input, NdArray teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            NdArray lossResult = lossFunction.Evaluate(functionStack.Forward(input), teach, out sumLoss);

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
            return BatchTrain(functionStack, NdArray.FromArray(input), NdArray.FromArray(teach), lossFunction, isUpdate);
        }

        //バッチで学習処理を行う
        public static double BatchTrain(FunctionStack functionStack, NdArray[] input, NdArray[] teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            double sumLoss;

            //Forwardのバッチを実行
            NdArray[] lossResult = lossFunction.Evaluate(functionStack.Forward(input), teach, out sumLoss);

            //Backwardのバッチを実行
            functionStack.Backward(lossResult);

            //更新
            if (isUpdate)
            {
                functionStack.Update();
            }

            return sumLoss;
        }

        //精度測定
        public static double Accuracy(FunctionStack functionStack, Array[] x, Array[] y)
        {
            return Accuracy(functionStack, NdArray.FromArray(x), NdArray.FromArray(y));
        }

        public static double Accuracy(FunctionStack functionStack, NdArray[] x, NdArray[] y)
        {
            int matchCount = 0;

            NdArray[] forwardResult = functionStack.Predict(x);

            for (int i = 0; i < x.Length; i++)
            {
                if (Array.IndexOf(forwardResult[i].Data, forwardResult[i].Data.Max()) == y[i].Data[0])
                {
                    matchCount++;
                }
            }

            return matchCount / (double)x.Length;
        }
    }
}
