using System;

namespace KelpNet
{
    public interface ICompressibleActivation<T> : IFunction<T> where T : unmanaged, IComparable<T>
    {
        Func<T, T> ForwardActivate { get; set; }
        Func<T, T, T> BackwardActivate { get; set; }
    }
}
