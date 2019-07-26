using System;

namespace KelpNet
{
    public class Sin<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Sin";

        public Sin(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> x)
        {
            RealArray<T> resultData = new T[x.DataLength];

            for (int i = 0; i < x.DataLength; i++)
            {
                resultData[i] = Math.Sin(x.Data[i]);
            }

            return new NdArray<T>(resultData, x.Shape, x.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                x.Grad[i] += Math.Cos(x.Data[i]) * y.Grad[i];
            }
        }
    }
}
