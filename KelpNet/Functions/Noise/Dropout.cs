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
        private readonly List<RealArray<T>> maskStack = new List<RealArray<T>>();

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.dropoutRatio = dropoutRatio;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private RealArray<T> MakeMask(int xLength)
        {
            RealArray<T> mask = new T[xLength];
            Real<T> scale = 1 / (1 - this.dropoutRatio);

            for (int i = 0; i < xLength; i++)
            {
                mask[i] = Mother<T>.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

            this.maskStack.Add(mask);

            return mask;
        }

        public NdArray<T> ForwardCpu(NdArray<T> x)
        {
            RealArray<T> result = new T[x.DataLength];
            RealArray<T> mask = MakeMask(x.Length);

            for (int i = 0; i < x.DataLength; i++)
            {
                result[i] = x.Data[i] * mask[i % x.Length];
            }

            return NdArray<T>.Convert(result, x.Shape, x.BatchCount, this);
        }

        public void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            RealArray<T> result = y.Grad.Clone();
            RealArray<T> mask = this.maskStack[this.maskStack.Count - 1];
            this.maskStack.RemoveAt(this.maskStack.Count - 1);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < mask.Length; i++)
                {
                    result[b * y.Length + i] *= mask[i];
                }
            }

            for (int i = 0; i < x.DataLength; i++)
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
