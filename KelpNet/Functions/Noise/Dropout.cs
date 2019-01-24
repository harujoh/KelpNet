using System;
using System.Collections.Generic;
using System.Linq;

namespace KelpNet
{
    [Serializable]
    public class Dropout<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Dropout";

        private readonly Real<T> dropoutRatio;
        private readonly List<Real<T>[]> maskStack = new List<Real<T>[]>();

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.dropoutRatio = dropoutRatio;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private Real<T>[] MakeMask(int xLength)
        {
            Real<T>[] mask = new Real<T>[xLength];
            Real<T> scale = 1 / (1 - this.dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother<T>.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

            this.maskStack.Add(mask);

            return mask;
        }

        public NdArray<T> ForwardCpu(NdArray<T> x)
        {
            Real<T>[] result = new Real<T>[x.Data.Length];
            Real<T>[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray<T>.Convert(result, x.Shape, x.BatchCount, this);
        }

        public void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            Real<T>[] result = y.Grad.ToArray();
            Real<T>[] mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

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

        //Predict時に何もしない
        public override NdArray<T> Predict(NdArray<T> input)
        {
            return input;
        }
    }
}
