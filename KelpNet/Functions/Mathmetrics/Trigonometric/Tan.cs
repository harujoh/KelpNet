using System;

namespace KelpNet
{
    public class Tan<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Tan";

        public Tan(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> x)
        {
            RealArray<T> resultData = new T[x.DataLength];

            for (int i = 0; i < x.DataLength; i++)
            {
                resultData[i] = Math.Tan(x.Data[i]);
            }

            return new NdArray<T>(resultData, x.Shape, x.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                Real<T> gx = Math.Cos(x.Data[i]);
                x.Grad[i] += 1 / (gx * gx) * y.Grad[i];
            }
        }
    }
}
