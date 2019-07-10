using System;

namespace KelpNet.CL
{
    public interface IParallelizable
    {
        bool IsParallel { get; set; }

        void InitParallel();
        bool SetParallel(bool enable);
    }
}
