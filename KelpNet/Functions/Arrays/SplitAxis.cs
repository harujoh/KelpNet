using System;
using System.Linq;

namespace KelpNet
{
    public class SplitAxis<T> : MultiOutputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "SplitAxis";
        public int Axis;
        public int[] Indices;

        public SplitAxis(int indices, int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Indices = new[] { indices };
            this.Axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public SplitAxis(int[] indices, int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Indices = indices.ToArray();
            this.Axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray<T>[] ForwardCpu(NdArray<T> x)
        {
            NdArray<T>[] resultArrays = NdArray<T>.Split(x, Indices, Axis);

            for (int i = 0; i < resultArrays.Length; i++)
            {
                resultArrays[i].ParentFunc = this;
            }

            return resultArrays;
        }

        private void BackwardCpu(NdArray<T>[] ys, NdArray<T> x)
        {
            NdArray<T> resultNdArray = ys[0].Clone();

            for (int i = 1; i < ys.Length; i++)
            {
                resultNdArray = NdArray<T>.Concatenate(resultNdArray, ys[i], Axis);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += resultNdArray.Grad[i];
            }
        }
    }
}
