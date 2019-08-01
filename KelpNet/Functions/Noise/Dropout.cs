using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "Dropout", Namespace = "KelpNet")]
    public class Dropout : SingleInputFunction
    {
        const string FUNCTION_NAME = "Dropout";

        [DataMember]
        public Real DropoutRatio;

        [DataMember]
        protected readonly List<Real[]> maskStack = new List<Real[]>();

        public Dropout(double dropoutRatio = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.DropoutRatio = dropoutRatio;
        }

        public Dropout(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        protected Real[] MakeMask(int xLength)
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

        public override NdArray SingleInputForward(NdArray x)
        {
            Real[] result = new Real[x.Data.Length];
            Real[] mask = MakeMask(x.Length);

            for (int i = 0; i < x.Data.Length; i++)
            {
                result[i] = x.Data[i] * mask[i % mask.Length];
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
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
