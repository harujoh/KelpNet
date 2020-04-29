﻿using System;
using System.Runtime.Serialization;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "TanhActivation", Namespace = "KelpNet")]
    public class TanhActivation<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "TanhActivation";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T> BackwardActivate { get; set; }

        public TanhActivation(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case TanhActivation<float> tanhActivationF:
                    tanhActivationF.SingleInputForward = tanhActivationF.NeedPreviousForwardCpu;
                    tanhActivationF.SingleOutputBackward = tanhActivationF.NeedPreviousBackwardCpu;
                    tanhActivationF.ForwardActivate = TanhActivationF.ForwardActivate;
                    tanhActivationF.BackwardActivate = TanhActivationF.BackwardActivate;
                    break;

                case TanhActivation<double> tanhActivationD:
                    tanhActivationD.SingleInputForward = tanhActivationD.NeedPreviousForwardCpu;
                    tanhActivationD.SingleOutputBackward = tanhActivationD.NeedPreviousBackwardCpu;
                    tanhActivationD.ForwardActivate = TanhActivationD.ForwardActivate;
                    tanhActivationD.BackwardActivate = TanhActivationD.BackwardActivate;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class TanhActivationD
#else
    public static class TanhActivationF
#endif
    {
        public static Real ForwardActivate(Real x)
        {
            return KelpMath.Tanh(x);
        }

        public static Real BackwardActivate(Real gy, Real y)
        {
            return gy * (1 - y * y);
        }
    }
}
