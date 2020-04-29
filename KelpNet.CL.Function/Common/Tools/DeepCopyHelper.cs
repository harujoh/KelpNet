using System;
using System.IO;
using System.Runtime.Serialization;

namespace KelpNet.CL
{
    //コチラを参考にしています
    //http://d.hatena.ne.jp/tekk/20100131/1264913887

    public static class DeepCopyHelper<T> where T : unmanaged, IComparable<T>
    {
        public static CopyType DeepCopy<CopyType>(CopyType target)
        {
            CopyType result;

            using (MemoryStream mem = new MemoryStream())
            {
                DataContractSerializer bf = new DataContractSerializer(typeof(Function<T>), ModelIO<T>.KnownTypes);

                try
                {
                    bf.WriteObject(mem, target);
                    mem.Position = 0;
                    result = (CopyType) bf.ReadObject(mem);
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
