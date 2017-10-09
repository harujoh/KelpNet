using System;
using KelpNet.Common.Functions;
using KelpNet.Common.Loss;

namespace KelpNet.Common.Tools
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        public static Real Train(FunctionStack functionStack, Array input, Array teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, new NdArray(input), new NdArray(teach), lossFunction, isUpdate);
        }

        //バッチで学習処理を行う
        public static Real Train(FunctionStack functionStack, Array[] input, Array[] teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, NdArray.FromArrays(input), NdArray.FromArrays(teach), lossFunction, isUpdate);
        }

        //バッチで学習処理を行う
        public static Real Train(FunctionStack functionStack, NdArray input, NdArray teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            Real sumLoss;

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

        //精度測定
        public static double Accuracy(FunctionStack functionStack, Array x, Array y)
        {
            return Accuracy(functionStack, new NdArray(x), new NdArray(y));
        }

        //精度測定
        public static double Accuracy(FunctionStack functionStack, Array[] x, Array[] y)
        {
            return Accuracy(functionStack, NdArray.FromArrays(x), NdArray.FromArrays(y));
        }

        public static double Accuracy(FunctionStack functionStack, NdArray x, NdArray y)
        {
            double matchCount = 0;

            NdArray forwardResult = functionStack.Predict(x);

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 0; i < forwardResult.Length; i++)
                {
                    if (maxval < forwardResult.Data[i + b * forwardResult.Length])
                    {
                        maxval = forwardResult.Data[i + b * forwardResult.Length];
                        maxindex = i;
                    }
                }

                if (maxindex == (int)y.Data[b * y.Length])
                {
                    matchCount++;
                }
            }

            return matchCount / x.BatchCount;
        }
    }
}
