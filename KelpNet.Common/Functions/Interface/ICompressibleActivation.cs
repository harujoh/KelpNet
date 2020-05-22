using System;

namespace KelpNet.CPU
{
    public interface ICompressibleActivation<T> : IFunction<T> where T : unmanaged, IComparable<T>
    {
        Func<T, T> ForwardActivate { get; set; }
        Func<T, T, T, T> BackwardActivate { get; set; }
    }
}
