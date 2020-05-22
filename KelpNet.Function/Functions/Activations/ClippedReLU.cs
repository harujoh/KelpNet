using System;
using System.Runtime.Serialization;
using KelpNet.CPU;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    [DataContract(Name = "ClippedReLU", Namespace = "KelpNet")]
    public class ClippedReLU<T> : SingleInputFunction<T>, ICompressibleActivation<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "ClippedReLU";

        public Func<T, T> ForwardActivate { get; set; }
        public Func<T, T, T, T> BackwardActivate { get; set; }

        [DataMember]
        public T Cap;

        public ClippedReLU(string name, string[] inputNames, string[] outputNames) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public ClippedReLU(T? cap = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Cap = cap ?? (TVal<T>)20;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case ClippedReLU<float> clippedReLuF:
                    clippedReLuF.SingleInputForward = clippedReLuF.NeedPreviousForwardCpu;
                    clippedReLuF.SingleOutputBackward = clippedReLuF.NeedPreviousBackwardCpu;
                    clippedReLuF.ForwardActivate = (x) => ClippedReLUF.ForwardActivate(x, clippedReLuF.Cap);
                    clippedReLuF.BackwardActivate = (gy, y, x) => ClippedReLUF.BackwardActivate(gy, y, x, clippedReLuF.Cap);
                    break;

                case ClippedReLU<double> clippedReLuD:
                    clippedReLuD.SingleInputForward = clippedReLuD.NeedPreviousForwardCpu;
                    clippedReLuD.SingleOutputBackward = clippedReLuD.NeedPreviousBackwardCpu;
                    clippedReLuD.ForwardActivate = (x) => ClippedReLUD.ForwardActivate(x, clippedReLuD.Cap);
                    clippedReLuD.BackwardActivate = (gy, y, x) => ClippedReLUD.BackwardActivate(gy, y, x, clippedReLuD.Cap);
                    break;
            }
        }
    }

    //ReLU6はClippedReLUの値が6で固定された物
    [DataContract(Name = "ReLU6", Namespace = "KelpNet")]
    public class ReLU6<T> : ClippedReLU<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "ReLU6";

        public ReLU6(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base((TVal<T>)6, name, inputNames, outputNames) { }
    }
#endif

#if DOUBLE
    public static class ClippedReLUD
#else
    public static class ClippedReLUF
#endif
    {
        public static Real ForwardActivate(Real x, Real cap)
        {
            return x < 0 ? 0 : x > cap ? cap : x;
        }

        public static Real BackwardActivate(Real gy, Real y, Real x, Real cap)
        {
            return 0 < x && x < cap ? gy : 0;
        }
    }

}
