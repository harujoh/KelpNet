using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace CaffemodelLoader
{
    [Serializable]
    public class FunctionSet : FunctionStack
    {
        const string FUNCTION_NAME = "FunctionSet";

        public string[] InputNames;
        public NdArray[] Result;

        public FunctionSet(Function function, string[] inputNames, string name= FUNCTION_NAME) : base(new[] { function }, name)
        {
            this.InputNames = inputNames.ToArray();
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
