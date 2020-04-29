using System;
using System.Collections.Generic;

namespace KelpNet.CL
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack<T> : CPU.FunctionStack<T> where T : unmanaged, IComparable<T>
    {
        //コンストラクタ
        public FunctionStack(Function<T>[] functions, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(functions, name, inputNames, outputNames)
        {
        }

        public FunctionStack(Function<T> function, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(function, name, inputNames, outputNames)
        {
        }

        public FunctionStack(params Function<T>[] functions) : base(functions)
        {
        }

        public override void Compress()
        {
            List<Function<T>> functionList = new List<Function<T>>(Functions);

            //層を圧縮
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is ICompressibleFunction<T> compressibleFunction)
                {
                    if (compressibleFunction.Activation == null && functionList[i + 1] is ICompressibleActivation<T> compressibleActivation)
                    {
                        compressibleFunction.Activation = compressibleActivation;
                        compressibleFunction.SetParallel(compressibleFunction.IsParallel);
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            this.Functions = functionList.ToArray();
        }
    }
}
