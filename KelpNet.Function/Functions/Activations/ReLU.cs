using System;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "ReLU", Namespace = "KelpNet")]
    public class ReLU<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "ReLU";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T, T> BackwardActivate { get; set; }

        public ReLU(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case ReLU<float> reluF:
                    reluF.SingleInputForward = reluF.NeedPreviousForwardCpu;
                    reluF.SingleOutputBackward = reluF.NeedPreviousBackwardCpu;
                    reluF.ForwardActivate = ReLUF.ForwardActivate;
                    reluF.BackwardActivate = ReLUF.BackwardActivate;
                    break;

                case ReLU<double> reluD:
                    reluD.SingleInputForward = reluD.NeedPreviousForwardCpu;
                    reluD.SingleOutputBackward = reluD.NeedPreviousBackwardCpu;
                    reluD.ForwardActivate = ReLUD.ForwardActivate;
                    reluD.BackwardActivate = ReLUD.BackwardActivate;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class ReLUD
#else
    public static class ReLUF
#endif
    {
        public static Real ForwardActivate(Real x)
        {
            return x < 0 ? 0 : x;
        }

        public static Real BackwardActivate(Real gy, Real y, Real x)
        {
            return y <= 0 ? 0 : gy;
        }
    }
}
