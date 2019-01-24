using System;
using System.IO;
using System.Runtime.Serialization;

namespace KelpNet.Tools
{
    public class ModelIO<T> where T : unmanaged, IComparable<T>
    {
        public static void Save(FunctionStack<T> functionStack, string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();

            using (Stream stream = File.OpenWrite(fileName))
            {
                bf.Serialize(stream, functionStack);
            }
        }

        public static FunctionStack<T> Load(string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();
            FunctionStack<T> result;

            using (Stream stream = File.OpenRead(fileName))
            {
                result = (FunctionStack<T>)bf.Deserialize(stream);
            }

            foreach (Function<T> function in result.Functions)
            {
                function.ResetState();

                for (int i = 0; i < function.Optimizers.Length; i++)
                {
                    function.Optimizers[i].ResetParams();
                }
            }

            return result;
        }
    }
}
