using System;

namespace KelpNet
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArray<T>の型変換を担う
    public class Trainer<T> where T : unmanaged, IComparable<T>
    {
        //バッチで学習処理を行う
        public static Real<T> Train(FunctionStack<T> functionStack, Array[] input, Array[] teach, LossFunction<T> lossFunction, bool isUpdate = true)
        {
            return Train(functionStack, NdArray<T>.FromArrays(input), NdArray<T>.FromArrays(teach), lossFunction, isUpdate);
        }

        //バッチで学習処理を行う
        public static Real<T> Train(FunctionStack<T> functionStack, NdArray<T> input, NdArray<T> teach, LossFunction<T> lossFunction, bool isUpdate = true)
        {
            //結果の誤差保存用
            NdArray<T>[] result = functionStack.Forward(input);
            Real<T> sumLoss = lossFunction.Evaluate(result, teach);

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
        public static Real<T> Accuracy(FunctionStack<T> functionStack, Array[] x, Array[] y)
        {
            return Accuracy(functionStack, NdArray<T>.FromArrays(x), NdArray<T>.FromArrays(y));
        }

        public static Real<T> Accuracy(FunctionStack<T> functionStack, NdArray<T> x, NdArray<T> y)
        {
            Real<T> matchCount = 0;

            NdArray<T> forwardResult = functionStack.Predict(x)[0];

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real<T> maxval = forwardResult.Data[b * forwardResult.Length];
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
