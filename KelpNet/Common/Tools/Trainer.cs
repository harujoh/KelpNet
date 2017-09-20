using System;
using KelpNet.Common.Loss;

namespace KelpNet.Common.Tools
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        public static Real Train(FunctionStack functionStack, Array input, Array teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, new BatchArray(input), new BatchArray(teach), lossFunction, isUpdate);
        }

        //バッチで学習処理を行う
        public static Real Train(FunctionStack functionStack, Array[] input, Array[] teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, BatchArray.FromArrays(input), BatchArray.FromArrays(teach), lossFunction, isUpdate);
        }

        //バッチで学習処理を行う
        public static Real Train(FunctionStack functionStack, BatchArray input, BatchArray teach, ILossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            Real sumLoss;

            //Forwardのバッチを実行
            BatchArray lossResult = lossFunction.Evaluate(functionStack.Forward(input), teach, out sumLoss);

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
            return Accuracy(functionStack, new BatchArray(x), new BatchArray(y));
        }

        //精度測定
        public static double Accuracy(FunctionStack functionStack, Array[] x, Array[] y)
        {
            return Accuracy(functionStack, BatchArray.FromArrays(x), BatchArray.FromArrays(y));
        }

        public static double Accuracy(FunctionStack functionStack, BatchArray x, BatchArray y)
        {
            double matchCount = 0;

            BatchArray forwardResult = functionStack.Predict(x);

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
