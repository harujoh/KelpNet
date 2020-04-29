using System;
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
    [DataContract(Name = "Sigmoid", Namespace = "KelpNet")]
    public class Sigmoid<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Sigmoid";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T> BackwardActivate { get; set; }

        public Sigmoid(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Sigmoid<float> sigmoidF:
                    sigmoidF.SingleInputForward = sigmoidF.NeedPreviousForwardCpu;
                    sigmoidF.SingleOutputBackward = sigmoidF.NeedPreviousBackwardCpu;
                    sigmoidF.ForwardActivate = SigmoidF.ForwardActivate;
                    sigmoidF.BackwardActivate = SigmoidF.BackwardActivate;
                    break;

                case Sigmoid<double> sigmoidD:
                    sigmoidD.SingleInputForward = sigmoidD.NeedPreviousForwardCpu;
                    sigmoidD.SingleOutputBackward = sigmoidD.NeedPreviousBackwardCpu;
                    sigmoidD.ForwardActivate = SigmoidD.ForwardActivate;
                    sigmoidD.BackwardActivate = SigmoidD.BackwardActivate;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SigmoidD
#else
    public static class SigmoidF
#endif
    {
        public static Real ForwardActivate(Real x)
        {
            return KelpMath.Tanh(x * 0.5f) * 0.5f + 0.5f;
        }

        public static Real BackwardActivate(Real gy, Real y)
        {
            return gy * y * (1.0f - y);
        }
    }
}
