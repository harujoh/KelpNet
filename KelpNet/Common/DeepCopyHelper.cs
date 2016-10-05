using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace KelpNet.Common
{
    //コチラを参考にしています
    //http://d.hatena.ne.jp/tekk/20100131/1264913887

    public static class DeepCopyHelper
    {
        public static T DeepCopy<T>(T target)
        {
            BinaryFormatter bf = new BinaryFormatter();
            MemoryStream mem = new MemoryStream();

            T result;
            try
            {
                bf.Serialize(mem, target);
                mem.Position = 0;
                result = (T)bf.Deserialize(mem);
            }
            finally
            {
                mem.Close();
            }

            return result;

        }
    }
}
