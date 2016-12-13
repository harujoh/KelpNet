using System;
using System.Collections.Generic;
using KelpNet.Common;
using System.Threading.Tasks;

namespace KelpNet.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousDataFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<NdArray[]> _prevInput = new List<NdArray[]>();
        private readonly List<NdArray[]> _prevOutput = new List<NdArray[]>();

        protected abstract NdArray NeedPreviousForward(NdArray x);
        protected abstract NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput);

        protected NeedPreviousDataFunction(string name, bool isParallel = true, int inputCount = 0, int oututCount = 0) : base(name, isParallel, inputCount, oututCount)
        {
        }

        protected override NdArray ForwardSingle(NdArray x)
        {
            this._prevInput.Add(new[] { x });
            NdArray result = this.NeedPreviousForward(x);
            this._prevOutput.Add(new[] { result });

            return result;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            this._prevInput.Add(x);

            NdArray[] prevoutput = new NdArray[x.Length];

            if (IsParallel)
            {
                Parallel.For(0, x.Length, i =>
                {
                    prevoutput[i] = this.NeedPreviousForward(x[i]);
                });
            }
            else
            {
                for (int i = 0; i < x.Length; i++)
                {
                    prevoutput[i] = this.NeedPreviousForward(x[i]);
                }
            }

            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        protected override NdArray BackwardSingle(NdArray gy)
        {
            NdArray prevInput = this._prevInput[this._prevInput.Count - 1][0];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            NdArray prevOutput = this._prevOutput[this._prevOutput.Count - 1][0];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput, prevOutput);
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            NdArray[] prevInput = this._prevInput[this._prevInput.Count - 1];
            this._prevInput.RemoveAt(this._prevInput.Count - 1);

            NdArray[] prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            NdArray[] result = new NdArray[gy.Length];

            if (IsParallel)
            {
                Parallel.For(0, gy.Length, i =>
                {
                    result[i] = this.NeedPreviousBackward(gy[i], prevInput[i], prevOutput[i]);
                });
            }
            else
            {
                for (int i = 0; i < gy.Length; i++)
                {
                    result[i] = this.NeedPreviousBackward(gy[i], prevInput[i], prevOutput[i]);
                }
            }

            return result;
        }

        public override NdArray Predict(NdArray input)
        {
            return this.NeedPreviousForward(input);
        }

        public override NdArray[] Predict(NdArray[] x)
        {
            NdArray[] prevoutput = new NdArray[x.Length];

            if (IsParallel)
            {
                Parallel.For(0, x.Length, i =>
                {
                    prevoutput[i] = this.NeedPreviousForward(x[i]);
                });
            }
            else
            {
                for (int i = 0; i < x.Length; i++)
                {
                    prevoutput[i] = this.NeedPreviousForward(x[i]);
                }
            }

            return prevoutput;
        }
    }
}
