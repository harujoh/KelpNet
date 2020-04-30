using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class Concat<T> : MultiInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Concat";
        public int Axis;

        private readonly List<int[]> prevInputSections = new List<int[]>();

        public Concat(int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.MultiInputForward = this.ConcatForward;

            switch (this)
            {
                case Concat<float> concatF:
                    concatF.MultiOutputBackward = (y, xs) => ConcatF.MultiOutputBackward(y, xs, concatF.Axis, concatF.prevInputSections);
                    break;

                case Concat<double> concatD:
                    concatD.MultiOutputBackward = (y, xs) => ConcatD.MultiOutputBackward(y, xs, concatD.Axis, concatD.prevInputSections);
                    break;
            }
        }

        protected NdArray<T>[] ConcatForward(params NdArray<T>[] xs)
        {
            int[] sections = new int[xs.Length - 1];
            int sizeOffset = xs[0].Shape[Axis];

            NdArray<T> resultNdArray = xs[0].Clone();

            for (int i = 1; i < xs.Length; i++)
            {
                //BackwardのSplitで使用しないため最後のshapeを保存しないロジックになっている
                sections[i - 1] = sizeOffset;
                sizeOffset += xs[i].Shape[Axis];

                resultNdArray = NdArray.Concatenate(resultNdArray, xs[i], Axis);
            }

            resultNdArray.ParentFunc = this;

            prevInputSections.Add(sections);

            return new[] { resultNdArray };
        }
    }
#endif

#if DOUBLE
    public static class ConcatD
#else
    public static class ConcatF
#endif
    {
        public static void MultiOutputBackward(NdArray<Real> y, NdArray<Real>[] xs, int axis, List<int[]> prevInputSections)
        {
            int[] prevInputShapes = prevInputSections[prevInputSections.Count - 1];
            prevInputSections.RemoveAt(prevInputSections.Count - 1);

            NdArray<Real>[] result = NdArray.Split(y, prevInputShapes, axis);

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Grad.Length; j++)
                {
                    xs[i].Grad[j] += result[i].Grad[j];
                }
            }
        }
    }
}
