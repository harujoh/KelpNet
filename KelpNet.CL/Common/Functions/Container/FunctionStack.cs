using System;
using System.Collections.Generic;

namespace KelpNet.CL
{
    //層を積み上げるこのライブラリのメインとなるクラス
    //一回のForward、Backward、Updateで同時に実行される関数の集まり
    [Serializable]
    public class FunctionStack : CPU.FunctionStack
    {
        //コンストラクタ
        public FunctionStack(Function[] functions, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(functions, name, inputNames, outputNames)
        {
        }

        public FunctionStack(Function function, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(function, name, inputNames, outputNames)
        {
        }

        public FunctionStack(params Function[] functions) : base(functions)
        {
        }

        public override void Compress()
        {
            List<Function> functionList = new List<Function>(Functions);

            //層を圧縮
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is ICompressibleFunction compressibleFunction)
                {
                    if (compressibleFunction.Activation == null && functionList[i + 1] is ICompressibleActivation compressibleActivation)
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
