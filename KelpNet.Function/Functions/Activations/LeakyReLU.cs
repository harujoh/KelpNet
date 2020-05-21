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
    [DataContract(Name = "LeakyReLU", Namespace = "KelpNet")]
    public class LeakyReLU<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "LeakyReLU";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T> BackwardActivate { get; set; }

        [DataMember]
        public T Slope;

        public LeakyReLU(string name, string[] inputNames, string[] outputNames) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public LeakyReLU(T? slope = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Slope = slope??(TVal<T>)0.2;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case LeakyReLU<float> leakyReLuF:
                    leakyReLuF.SingleInputForward = leakyReLuF.NeedPreviousForwardCpu;
                    leakyReLuF.SingleOutputBackward = leakyReLuF.NeedPreviousBackwardCpu;
                    leakyReLuF.ForwardActivate = (x) => LeakyReLUF.ForwardActivate(x, leakyReLuF.Slope);
                    leakyReLuF.BackwardActivate = (gy, y) => LeakyReLUF.BackwardActivate(gy, y, leakyReLuF.Slope);
                    break;

                case LeakyReLU<double> leakyReLuD:
                    leakyReLuD.SingleInputForward = leakyReLuD.NeedPreviousForwardCpu;
                    leakyReLuD.SingleOutputBackward = leakyReLuD.NeedPreviousBackwardCpu;
                    leakyReLuD.ForwardActivate = (x) => LeakyReLUD.ForwardActivate(x, leakyReLuD.Slope);
                    leakyReLuD.BackwardActivate = (gy, y) => LeakyReLUD.BackwardActivate(gy, y, leakyReLuD.Slope);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class LeakyReLUD
#else
    public static class LeakyReLUF
#endif
    {
        public static Real ForwardActivate(Real x, Real slope)
        {
            return x < 0 ? x * slope : x;
        }

        public static Real BackwardActivate(Real gy, Real y, Real slope)
        {
            return y <= 0 ? y * slope : gy;
        }
    }

}
