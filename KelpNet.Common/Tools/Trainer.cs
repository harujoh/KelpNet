using System;
using KelpNet.CPU;

namespace KelpNet
{
    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        //バッチで学習処理を行う
        public static T Train<T, LabelType>(FunctionStack<T> functionStack, T[][] input, LabelType[][] teach, LossFunction<T, LabelType> lossFunction, Optimizer<T> optimizer = null) where T : unmanaged, IComparable<T> where LabelType : unmanaged, IComparable<LabelType>
        {
            return Train(functionStack, NdArray.FromArrays(input), NdArray.FromArrays(teach), lossFunction, optimizer);
        }

        public static T Train<T>(FunctionStack<T> functionStack, TestDataSet<T> dataSet, LossFunction<T, int> lossFunction, Optimizer<T> optimizer = null) where T : unmanaged, IComparable<T>
        {
            return Train(functionStack, dataSet.Data, dataSet.Label, lossFunction, optimizer);
        }

        public static T Train<T, LabelType>(FunctionStack<T> functionStack, NdArray<T> input, NdArray<LabelType> teach, LossFunction<T, LabelType> lossFunction, Optimizer<T> optimizer = null) where T : unmanaged, IComparable<T> where LabelType : unmanaged, IComparable<LabelType>
        {
            optimizer?.SetUp(functionStack);

            //結果の誤差保存用
            NdArray<T> result = functionStack.Forward(input)[0];
            T loss = lossFunction.Evaluate(result, teach);

            //Backwardのバッチを実行
            functionStack.Backward(result);

            //更新
            optimizer?.Update();

            return loss;
        }

        //精度測定
        public static T Accuracy<T>(FunctionStack<T> functionStack, T[][] x, int[][] y) where T : unmanaged, IComparable<T>
        {
            return Accuracy(functionStack, NdArray.FromArrays(x), NdArray.FromArrays(y));
        }

        public static T Accuracy<T>(FunctionStack<T> functionStack, TestDataSet<T> dataSet) where T : unmanaged, IComparable<T>
        {
            return Accuracy(functionStack, dataSet.Data, dataSet.Label);
        }

        public static T Accuracy<T>(FunctionStack<T> functionStack, NdArray<T> x, NdArray<int> y) where T : unmanaged, IComparable<T>
        {
            int matchCount = 0;

            NdArray<T> forwardResult = functionStack.Predict(x)[0];

            for (int b = 0; b < x.BatchCount; b++)
            {
                T maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 1; i < forwardResult.Length; i++)
                {
                    if (maxval.CompareTo(forwardResult.Data[b * forwardResult.Length + i]) < 0)
                    {
                        maxval = forwardResult.Data[b * forwardResult.Length + i];
                        maxindex = i;
                    }
                }

                if (maxindex == y.Data[b * y.Length])
                {
                    matchCount++;
                }
            }

            return (TVal<T>)matchCount / (TVal<T>)x.BatchCount;
        }

        //精度測定
        public static T Accuracy<T>(FunctionStack<T> functionStack, T[][] x, int[][] y, LossFunction<T, int> lossFunction, out T loss) where T : unmanaged, IComparable<T>
        {
            return Accuracy(functionStack, NdArray.FromArrays(x), NdArray.FromArrays(y), lossFunction, out loss);
        }

        public static T Accuracy<T>(FunctionStack<T> functionStack, TestDataSet<T> dataSet, LossFunction<T, int> lossFunction, out T loss) where T : unmanaged, IComparable<T>
        {
            return Accuracy(functionStack, dataSet.Data, dataSet.Label, lossFunction, out loss);
        }

        public static T Accuracy<T>(FunctionStack<T> functionStack, NdArray<T> x, NdArray<int> y, LossFunction<T, int> lossFunction, out T loss) where T : unmanaged, IComparable<T>
        {
            int matchCount = 0;

            NdArray<T> forwardResult = functionStack.Predict(x)[0];
            loss = lossFunction.Evaluate(forwardResult, y);

            for (int b = 0; b < x.BatchCount; b++)
            {
                T maxval = forwardResult.Data[b * forwardResult.Length];
                int maxindex = 0;

                for (int i = 1; i < forwardResult.Length; i++)
                {
                    if (maxval.CompareTo(forwardResult.Data[b * forwardResult.Length + i]) < 0)
                    {
                        maxval = forwardResult.Data[b * forwardResult.Length + i];
                        maxindex = i;
                    }
                }

                if (maxindex == y.Data[b * y.Length])
                {
                    matchCount++;
                }
            }

            return (TVal<T>)matchCount / (TVal<T>)x.BatchCount;
        }

    }
}
