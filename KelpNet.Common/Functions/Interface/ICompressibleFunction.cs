using System;

namespace KelpNet
{
    public interface ICompressibleFunction<T> : IFunction<T> where T : unmanaged, IComparable<T>
    {
        ICompressibleActivation<T> Activation { get; set; }
    }
}
