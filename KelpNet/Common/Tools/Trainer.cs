using System;
using KelpNet.CPU;

namespace KelpNet
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        //バッチで学習処理を行う
        public static Real Train(FunctionStack functionStack, Array[] input, Array[] teach, LossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, NdArray.FromArrays(input), NdArray.FromArrays(teach), lossFunction, isUpdate);
        }

        public static Real Train(FunctionStack functionStack, TestDataSet dataSet, LossFunction lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, dataSet.Data, dataSet.Label, lossFunction, isUpdate);
        }

        public static Real Train(FunctionStack functionStack, NdArray input, NdArray teach, LossFunction lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            NdArray result = functionStack.Forward(input)[0];
            Real sumLoss = lossFunction.Evaluate(result, teach);

            //Backwardのバッチを実行
            functionStack.Backward(result);

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
            return Accuracy(functionStack, NdArray.FromArrays(x), NdArray.FromArrays(y));
        }

        public static double Accuracy(FunctionStack functionStack, TestDataSet dataSet)
        {
            return Accuracy(functionStack, dataSet.Data, dataSet.Label);
        }

        public static double Accuracy(FunctionStack functionStack, NdArray x, NdArray y)
        {
            double matchCount = 0;

            NdArray forwardResult = functionStack.Predict(x)[0];

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 1; i < forwardResult.Length; i++)
                {
                    if (maxval < forwardResult.Data[b * forwardResult.Length + i])
                    {
                        maxval = forwardResult.Data[b * forwardResult.Length + i];
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
