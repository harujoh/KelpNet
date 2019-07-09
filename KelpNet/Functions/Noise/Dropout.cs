using System;
using System.Collections.Generic;
using System.Linq;

namespace KelpNet.CPU
{
    [Serializable]
    public class Dropout : SingleInputFunction, IParallelizable
    {
        const string FUNCTION_NAME = "Dropout";

        public Real DropoutRatio;
        private readonly List<Real[]> maskStack = new List<Real[]>();

        public bool IsParallel { get; set; }

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            IsParallel = false;
            this.DropoutRatio = dropoutRatio;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public bool SetParallel(bool enable)
        {
            return false;
        }

        public void InitParallel()
        {
        }

        private Real[] MakeMask(int xLength)
        {
            Real[] mask = new Real[xLength];
            Real scale = 1 / (1 - this.DropoutRatio);

            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = Mother.Dice.NextDouble() >= this.DropoutRatio ? scale : 0;
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
