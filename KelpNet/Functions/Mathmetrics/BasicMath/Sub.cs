using System;

namespace KelpNet
{
    public class Sub<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Sub";

        public Sub(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            Real<T>[] resultData = new Real<T>[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] - b.Data[i];
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i];
                b.Grad[i] -= y.Grad[i];
            }
        }
    }

    //右辺が定数
    public class SubConst<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "SubConst";

        public SubConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            Real<T>[] resultData = new Real<T>[a.Data.Length];
            Real<T> val = b.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] - val;
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i];
            }
        }
    }

    //左辺が定数
    public class ConstSub<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "ConstSub";

        public ConstSub(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            Real<T>[] resultData = new Real<T>[a.Data.Length];
            Real<T> val = a.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = val - b.Data[i];
            }

            return new NdArray<T>(resultData, b.Shape, b.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                b.Grad[i] -= y.Grad[i];
            }
        }
    }

}
