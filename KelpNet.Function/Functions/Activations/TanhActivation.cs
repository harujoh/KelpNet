using System;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "TanhActivation", Namespace = "KelpNet")]
    public class TanhActivation<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "TanhActivation";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T, T> BackwardActivate { get; set; }

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
            return Math.Tanh(x);
        }

        public static Real BackwardActivate(Real gy, Real y, Real x)
        {
            return gy * (1 - y * y);
        }
    }
}
