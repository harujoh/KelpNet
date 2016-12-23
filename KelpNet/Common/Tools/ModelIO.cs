using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using KelpNet.Functions;

namespace KelpNet.Common.Tools
{
    public class ModelIO
    {
        public static void Save(FunctionStack functionStack, string fileName)
        {
            BinaryFormatter bf = new BinaryFormatter();

            using (Stream stream = File.OpenWrite(fileName))
            {
                bf.Serialize(stream, functionStack);
            }
        }

        public static FunctionStack Load(string fileName)
        {
            BinaryFormatter bf = new BinaryFormatter();
            FunctionStack result;

            using (Stream stream = File.OpenRead(fileName))
            {
                result = (FunctionStack)bf.Deserialize(stream);
            }

            return result;
        }

    }
}
