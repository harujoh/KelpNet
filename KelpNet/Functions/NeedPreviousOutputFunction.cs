using System;
using System.Collections.Generic;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

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

        protected NeedPreviousOutputFunction(string name, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
        }

        protected override NdArray ForwardSingle(NdArray x)
        {
            var result = this.NeedPreviousForward(x);
            this._prevOutput.Add(new[] { result });

            return result;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] prevoutput = new NdArray[x.Length];

#if DEBUG
            for(int i = 0; i < x.Length; i ++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                prevoutput[i] = this.NeedPreviousForward(x[i]);
            }
#if !DEBUG
            );
#endif

            this._prevOutput.Add(prevoutput);

            return prevoutput;
        }

        protected override NdArray BackwardSingle(NdArray gy)
        {
            var prevOutput = this._prevOutput[this._prevOutput.Count-1][0];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevOutput);
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            var prevOutput = this._prevOutput[this._prevOutput.Count - 1];
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            NdArray[] result = new NdArray[gy.Length];

#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                result[i] = this.NeedPreviousBackward(gy[i], prevOutput[i]);
            }
#if !DEBUG
            );
#endif

            return result;
        }

        public override NdArray Predict(NdArray input)
        {
            return this.NeedPreviousForward(input);
        }

        public override NdArray[] Predict(NdArray[] x)
        {
            NdArray[] prevoutput = new NdArray[x.Length];
#if DEBUG
            for(int i = 0; i < x.Length; i ++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                prevoutput[i] = this.NeedPreviousForward(x[i]);
            }
#if !DEBUG
            );
#endif
            return prevoutput;
        }
    }
}
