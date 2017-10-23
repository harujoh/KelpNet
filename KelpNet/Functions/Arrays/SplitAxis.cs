using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    public class SplitAxis : MultiOutputFunction
    {
        const string FUNCTION_NAME = "SplitAxis";
        private int _axis;
        private int[] _indices;

        public SplitAxis(int[] indices, int axis, string name = FUNCTION_NAME) : base(name)
        {
            this._indices = indices.ToArray();
            this._axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public NdArray[] ForwardCpu(NdArray x)
        {
            NdArray[] resultArays = NdArray.Split(x, _indices, _axis);

            foreach (var resultArray in resultArays)
            {
                resultArray.ParentFunc = this;
            }

            return resultArays;
        }

        public void BackwardCpu(NdArray[] ys, NdArray x)
        {
            NdArray resultNdArray = ys[0].Clone();

            for (int i = 1; i < ys.Length; i++)
            {
                resultNdArray = NdArray.Concatenate(resultNdArray, ys[i], _axis);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += resultNdArray.Grad[i];
            }
        }
    }
}
