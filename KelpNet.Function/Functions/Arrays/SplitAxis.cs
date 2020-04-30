using System;
using System.Linq;
using System.Runtime.Serialization;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class SplitAxis<T> : MultiOutputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "SplitAxis";

        public int Axis;
        public int[] Indices;

        public SplitAxis(int[] indices, int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Indices = indices.ToArray();
            this.Axis = axis;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.SingleInputForward = this.SplitAxisForward;

            switch (this)
            {
                case SplitAxis<float> splitAxisF:
                    splitAxisF.MultiOutputBackward = (ys, x) => SplitAxisF.MultiOutputBackward(ys, x, splitAxisF.Axis);
                    break;
                case SplitAxis<double> splitAxisD:
                    splitAxisD.MultiOutputBackward = (ys, x) => SplitAxisD.MultiOutputBackward(ys, x, splitAxisD.Axis);
                    break;
            }
        }

        protected NdArray<T>[] SplitAxisForward(NdArray<T> x)
        {
            NdArray<T>[] resultArrays = NdArray.Split(x, Indices, Axis);

            for (int i = 0; i < resultArrays.Length; i++)
            {
                resultArrays[i].ParentFunc = this;
            }

            return resultArrays;
        }

    }
#endif

#if DOUBLE
    public static class SplitAxisD
#else
    public static class SplitAxisF
#endif
    {
        public static void MultiOutputBackward(NdArray<Real>[] ys, NdArray<Real> x, int axis)
        {
            NdArray<Real> resultNdArray = ys[0].Clone();

            for (int i = 1; i < ys.Length; i++)
            {
                resultNdArray = NdArray.Concatenate(resultNdArray, ys[i], axis);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += resultNdArray.Grad[i];
            }
        }
    }
}
