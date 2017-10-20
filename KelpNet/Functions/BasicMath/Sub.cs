﻿using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.BasicMath
{
    public class Sub : DualInputFunction
    {
        private const string FUNCTION_NAME = "Sub";

        public Sub(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] - b.Data[i];
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i];
                b.Grad[i] -= y.Grad[i];
            }
        }
    }

    //右辺が定数
    public class SubConst : DualInputFunction
    {
        private const string FUNCTION_NAME = "SubConst";

        public SubConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] - b.Data[0];
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i];
            }
        }
    }

    //左辺が定数
    public class ConstSub : DualInputFunction
    {
        private const string FUNCTION_NAME = "ConstSub";

        public ConstSub(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[0] - b.Data[i];
            }

            return new NdArray(resultData, this);
        }

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                b.Grad[i] -= y.Grad[i];
            }
        }
    }

}