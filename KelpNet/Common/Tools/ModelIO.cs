using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;

namespace KelpNet
{
    public class ModelIO
    {
        public static void Save(Function function, string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();

            //ZIP書庫を作成
            if (File.Exists(fileName))
            {
                File.Delete(fileName);
            }

            using (ZipArchive zipArchive = ZipFile.Open(fileName, ZipArchiveMode.Create))
            {
                ZipArchiveEntry entry = zipArchive.CreateEntry("Function");
                using (Stream stream = entry.Open())
                {
                    bf.Serialize(stream, function);
                }
            }
        }

        public static Function Load(string fileName)
        {
            NetDataContractSerializer bf = new NetDataContractSerializer();

            ZipArchiveEntry zipData = ZipFile.OpenRead(fileName).GetEntry("Function");
            Function result = (Function)bf.Deserialize(zipData.Open());

            if (result is FunctionStack functionStack)
            {
                InitFunctionStack(functionStack);
            }
            else if (result is FunctionDictionary functionDictionary)
            {
                foreach (FunctionStack functionBlock in functionDictionary.FunctionBlocks)
                {
                    InitFunctionStack(functionBlock);
                }
            }

            return result;
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