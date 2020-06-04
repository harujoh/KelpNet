using System;
using System.IO;
using System.Runtime.Serialization;

namespace KelpNet.CPU
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
                DataContractSerializer bf = new DataContractSerializer(typeof(Function<T>), new DataContractSerializerSettings { KnownTypes = ModelIO<T>.KnownTypes, PreserveObjectReferences = true });

                try
                {
                    bf.WriteObject(mem, target);
                    mem.Position = 0;
                    result = (CopyType)bf.ReadObject(mem);
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
