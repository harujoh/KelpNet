using System;
using System.Runtime.Serialization;
using KelpNet.CPU;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
#if !DOUBLE
    [DataContract(Name = "Mish", Namespace = "KelpNet")]
    public class Mish<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Mish";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T, T> BackwardActivate { get; set; }

        public Mish(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Mish<float> mishF:
                    mishF.SingleInputForward = mishF.NeedPreviousForwardCpu;
                    mishF.SingleOutputBackward = mishF.NeedPreviousBackwardCpu;
                    mishF.ForwardActivate = MishF.ForwardActivate;
                    mishF.BackwardActivate = MishF.BackwardActivate;
                    break;

                case Mish<double> mishD:
                    mishD.SingleInputForward = mishD.NeedPreviousForwardCpu;
                    mishD.SingleOutputBackward = mishD.NeedPreviousBackwardCpu;
                    mishD.ForwardActivate = (x) => MishD.ForwardActivate(x);
                    mishD.BackwardActivate = (gy, y, x) => MishD.BackwardActivate(gy, y, x);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class MishD
#else
    public static class MishF
#endif
    {
        public static Real ForwardActivate(Real x)
        {
            return x * Math.Tanh(Softplus(x));
        }

        public static Real BackwardActivate(Real gy, Real y, Real x)
        {
            Real sp = Softplus(x);
            Real tsp = Math.Tanh(sp);
            Real grad_tsp = (1 - tsp * tsp) * (1 - Math.Exp(-sp));

            return gy * (x * grad_tsp + tsp);
        }

        static Real Softplus(Real x)
        {
            if (x > 20) return x;
            if (x < -20) return Math.Exp(x);
            return Math.Log(Math.Exp(x) + 1);
        }
    }
}
