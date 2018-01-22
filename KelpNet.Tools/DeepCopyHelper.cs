using System.IO;
using System.Runtime.Serialization;

namespace KelpNet.Tools
{
    //コチラを参考にしています
    //http://d.hatena.ne.jp/tekk/20100131/1264913887

    public static class DeepCopyHelper
    {
        public static T DeepCopy<T>(T target)
        {
            T result;

            using (MemoryStream mem = new MemoryStream())
            {
                NetDataContractSerializer bf = new NetDataContractSerializer();

                try
                {
                    bf.Serialize(mem, target);
                    mem.Position = 0;
                    result = (T) bf.Deserialize(mem);
                }
                finally
                {
                    mem.Close();
                }
            }

            return result;
        }
    }
}
