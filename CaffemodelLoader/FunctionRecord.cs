using System;
using System.Linq;
using KelpNet.Common.Functions;

namespace CaffemodelLoader
{
    [Serializable]
    public class FunctionRecord : FunctionStack
    {
        const string FUNCTION_NAME = "FunctionRecord";

        public string[] InputNames;
        public string[] OutputNames;

        public FunctionRecord(Function function, string[] inputNames, string[] outputNames, string name= FUNCTION_NAME) : base(new[] { function }, name)
        {
            this.InputNames = inputNames.ToArray();
            this.OutputNames = outputNames.ToArray();
        }
    }
}
