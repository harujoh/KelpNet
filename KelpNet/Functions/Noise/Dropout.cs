using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Noise
{
    [Serializable]
    public class Dropout : SingleInputFunction
    {
        const string FUNCTION_NAME = "Dropout";

        private readonly Real dropoutRatio;
        private readonly List<Real[]> maskStack = new List<Real[]>();

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.dropoutRatio = dropoutRatio;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private Real[] MakeMask(int xLength)
        {
            Real[] mask = new Real[xLength];
            Real scale = 1 / (1 - this.dropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= this.dropoutRatio ? scale : 0;
            }

            this.maskStack.Add(mask);

            return mask;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public void BackwardCpu(NdArray y, NdArray x)
        {
            Real[] result = y.Grad.ToArray();
            Real[] mask = this.maskStack[this.maskStack.Count - 1];
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
        public override NdArray Predict(NdArray input)
        {
            return input;
        }
    }
}
