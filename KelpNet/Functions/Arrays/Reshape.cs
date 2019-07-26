using System;

namespace KelpNet
{
    class Reshape<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Reshape";
        public int[] Shape;

        public Reshape(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Shape = shape;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        NdArray<T> ForwardCpu(NdArray<T> val)
        {
            NdArray<T> result = val.Clone();
            result.ParentFunc = this;
            result.Reshape(this.Shape);

            return result;
        }

        void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            y.Grad = x.Grad.Clone();
        }
    }
}
