using System.IO;
using System.Runtime.Serialization;

namespace KelpNet.CL
{
    public static class CLConverter
    {
        public static T Convert<T>(T target)
        {
            T result;

            using (MemoryStream mem = new MemoryStream())
            {
                DataContractSerializer cpuDCS = new DataContractSerializer(typeof(T), CPU.ModelIO.KnownTypes);
                DataContractSerializer clDCS = new DataContractSerializer(typeof(T), ModelIO.KnownTypes);

                try
                {
                    cpuDCS.WriteObject(mem, target);
                    mem.Position = 0;
                    result = (T)clDCS.ReadObject(mem);

                    if (result is IParallelizable parallelizableFunc)
                    {
                        parallelizableFunc.SetParallel(true);
                    }
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
