using System;
using System.Linq;

namespace KelpNet.Common.Functions.Container
{
    [Serializable]
    public class FunctionRecord : FunctionStack
    {
        const string FUNCTION_NAME = "FunctionRecord";

        public string[] InputNames;
        public string[] OutputNames;

        public FunctionRecord(Function function, string[] inputNames, string[] outputNames, string name= FUNCTION_NAME) : base(function)
        {
            this.Name = name;
            this.InputNames = inputNames.ToArray();
            this.OutputNames = outputNames.ToArray();
        }
    }
}
