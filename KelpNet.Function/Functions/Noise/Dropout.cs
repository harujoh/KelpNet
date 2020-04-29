using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "Dropout", Namespace = "KelpNet")]
    public class Dropout<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Dropout";

        [DataMember]
        public T DropoutRatio;

        [DataMember]
        protected readonly List<T[]> maskStack = new List<T[]>();

        public Dropout(string name, string[] inputNames, string[] outputNames) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            switch (this)
            {
                case Dropout<float> dropOutF:
                    dropOutF.DropoutRatio = (float)dropoutRatio;
                    break;

                case Dropout<double> dropOutD:
                    dropOutD.DropoutRatio = dropoutRatio;
                    break;
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            this.Predict = x => x; //Predict時は何もしない

            switch (this)
            {
                case Dropout<float> dropOutF:
                    dropOutF.SingleInputForward = x => DropOutF.SingleInputForward(x, dropOutF.DropoutRatio, dropOutF.maskStack, dropOutF);
                    dropOutF.SingleOutputBackward = (y, x) => DropOutF.SingleOutputBackward(y, x, dropOutF.maskStack);
                    break;

                case Dropout<double> dropOutD:
                    dropOutD.SingleInputForward = x => DropOutD.SingleInputForward(x, dropOutD.DropoutRatio, dropOutD.maskStack, dropOutD);
                    dropOutD.SingleOutputBackward = (y, x) => DropOutD.SingleOutputBackward(y, x, dropOutD.maskStack);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class DropOutD
#else
    public static class DropOutF
#endif
    {
        public static Real[] MakeMask(int xLength, Real dropoutRatio, List<Real[]> maskStack)
        {
            Real[] mask = new Real[xLength];
            Real scale = 1 / (1 - dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= dropoutRatio ? scale : 0;
            }

            maskStack.Add(mask);

            return mask;
        }

        public static NdArray<Real> SingleInputForward(NdArray<Real> x, Real dropoutRatio, List<Real[]> maskStack, IFunction<Real> dropOut)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length, dropoutRatio, maskStack);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, dropOut);
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, List<Real[]> maskStack)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = maskStack[maskStack.Count - 1];
            maskStack.RemoveAt(maskStack.Count - 1);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[b * y.Length + i] *= mask[i];
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += result[i];
            }
        }
    }
}
