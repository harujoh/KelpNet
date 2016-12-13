using System;
using System.Collections.Generic;
using KelpNet.Common;
using System.Threading.Tasks;

namespace KelpNet.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousOutputFunction : Function
    {
        //後入れ先出しリスト
        private readonly List<NdArray[]> _prevOutput = new List<NdArray[]>();

        protected abstract NdArray NeedPreviousForward(NdArray x);
        protected abstract NdArray NeedPreviousBackward(NdArray gy, NdArray prevOutput);

        protected NeedPreviousOutputFunction(string name, bool isParallel = true, int inputCount = 0, int oututCount = 0) : base(name, isParallel, inputCount, oututCount)
        {
        }

        protected override NdArray ForwardSingle(NdArray x)
        {
            NdArray result = this.NeedPreviousForward(x);
            this._prevOutput.Add(new[] { result });

            return result;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
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

            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        protected override NdArray BackwardSingle(NdArray gy)
        {
            NdArray prevOutput = this._prevOutput[this._prevOutput.Count - 1][0];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevOutput);
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            NdArray[] prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            NdArray[] result = new NdArray[gy.Length];

            if (IsParallel)
            {
                Parallel.For(0, gy.Length, i =>
                {
                    result[i] = this.NeedPreviousBackward(gy[i], prevOutput[i]);
                });
            }
            else
            {
                for (int i = 0; i < gy.Length; i++)
                {
                    result[i] = this.NeedPreviousBackward(gy[i], prevOutput[i]);
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
