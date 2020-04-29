using System;
using System.Collections.Generic;

namespace KelpNet
{
    public delegate void ActionOptional<T>(params NdArray<T>[] args) where T : unmanaged, IComparable<T>;
    public delegate NdArray<T>[] FunctionOptional<T>(params NdArray<T>[] args) where T : unmanaged, IComparable<T>;

    public interface IFunction<T> where T : unmanaged, IComparable<T>
    {
        string Name { get; set; }

        NdArray<T>[] Parameters { get; set; }

        List<NdArray<T>[]> PrevInputs { get; set; }
        List<NdArray<T>[]> UsedPrevInputs { get; set; }

        string[] InputNames { get; set; }
        string[] OutputNames { get; set; }

        FunctionOptional<T> Forward { get; set; }
        FunctionOptional<T> Predict { get; set; }
        ActionOptional<T> Backward { get; set; }
    }
}
