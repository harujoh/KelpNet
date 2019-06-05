using System.IO;
using System.Runtime.Serialization;

namespace KelpNet
{
    public class ModelIO
    {
        public static void Save(Function function, string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();

            using (Stream stream = File.OpenWrite(fileName))
            {
                bf.Serialize(stream, function);
            }
        }

        public static T Load<T>(string fileName) where T : Function
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();
            Function result;

            using (Stream stream = File.OpenRead(fileName))
            {
                result = (Function)bf.Deserialize(stream);
            }

            if (result is FunctionStack functionStack)
            {
                InitFunctionStack(functionStack);
            }
            else if(result is FunctionDictionary functionDictionary)
            {
                foreach (FunctionStack functionBlock in functionDictionary.FunctionBlocks)
                {
                    InitFunctionStack(functionBlock);
                }
            }

            return (T)result;
        }

        static void InitFunctionStack(FunctionStack functionStack)
        {
            foreach (Function function in functionStack.Functions)
            {
                function.ResetState();

                for (int i = 0; i < function.Optimizers.Length; i++)
                {
                    function.Optimizers[i].ResetParams();
                }

                if (function is IParallelizable)
                {
                    ((IParallelizable)function).CreateKernel();
                }
            }
        }
    }
}
