using System;

namespace KelpNet
{
    public class Div<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "Div";

        public Div(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            RealArray<T> resultData = new T[a.DataLength];

            for (int i = 0; i < a.DataLength; i++)
            {
                resultData[i] = a.Data[i] / b.Data[i];
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            for (int i = 0; i < y.DataLength; i++)
            {
                Real<T> gx = y.Grad[i] / b.Data[i];
                a.Grad[i] += gx;
                b.Grad[i] += -gx * a.Data[i] / b.Data[i];
            }
        }
    }

    //右辺が定数
    public class DivConst<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "DivConst";

        public DivConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            RealArray<T> resultData = new T[a.DataLength];
            Real<T> val = b.Data[0];

            for (int i = 0; i < a.DataLength; i++)
            {
                resultData[i] = a.Data[i] / val;
            }

            return new NdArray<T>(resultData, a.Shape, a.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            Real<T> val = b.Data[0];

            for (int i = 0; i < y.DataLength; i++)
            {
                a.Grad[i] += y.Grad[i] / val;
            }
        }
    }

    //左辺が定数
    public class ConstDiv<T> : DualInputFunction<T> where T : unmanaged, IComparable<T>
    {
        private const string FUNCTION_NAME = "ConstDiv";

        public ConstDiv(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray<T> ForwardCpu(NdArray<T> a, NdArray<T> b)
        {
            RealArray<T> resultData = new T[a.DataLength];
            Real<T> val = a.Data[0];

            for (int i = 0; i < a.DataLength; i++)
            {
                resultData[i] = val / b.Data[i];
            }

            return new NdArray<T>(resultData, b.Shape, b.BatchCount, this);
        }

        protected void BackwardCpu(NdArray<T> y, NdArray<T> a, NdArray<T> b)
        {
            Real<T> val = a.Data[0];

            for (int i = 0; i < y.DataLength; i++)
            {
                Real<T> gx = y.Grad[i] / b.Data[i];
                b.Grad[i] += -gx * val / b.Data[i];
            }
        }
    }

}
