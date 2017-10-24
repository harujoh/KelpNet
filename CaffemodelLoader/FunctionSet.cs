using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace CaffemodelLoader
{
    [Serializable]
    public class FunctionSet : FunctionStack
    {
        public string[] InputNames;
        public NdArray[] Result;

        public FunctionSet(Function function, string[] inputNames)
        {
            this.InputNames = inputNames.ToArray();
            Add(function);
        }

        public override NdArray[] Forward(params NdArray[] xs)
        {
            Result = base.Forward(xs);

            return Result;
        }

        public override NdArray[] Predict(params NdArray[] xs)
        {
            Result = base.Predict(xs);

            return Result;
        }
    }
}
