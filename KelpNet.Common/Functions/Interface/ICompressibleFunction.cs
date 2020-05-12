using System;

namespace KelpNet.CPU
{
    public interface ICompressibleFunction<T> : IFunction<T> where T : unmanaged, IComparable<T>
    {
        ICompressibleActivation<T> Activation { get; set; }
    }
}
