using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions
{
    //前回の入出力を自動的に扱うクラステンプレート
    [Serializable]
    public abstract class NeedPreviousDataFunction : Function
    {
        //後入れ先出しリスト
        private List<NdArray[]> _prevInput = new List<NdArray[]>();
        private List<NdArray[]> _prevOutput = new List<NdArray[]>();

        protected abstract NdArray NeedPreviousForward(NdArray x);
        protected abstract NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput);

        protected NeedPreviousDataFunction(string name) : base(name)
        {
        }

        protected override NdArray ForwardSingle(NdArray x)
        {
            this._prevInput.Add(new[] { x });
            this._prevOutput.Add(new[] { this.NeedPreviousForward(x) });

            return this._prevOutput.Last()[0];
        }


        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] prevInput = new NdArray[x.Length];
            for (int i = 0; i < prevInput.Length; i++)
            {
                prevInput[i] = new NdArray(x[i]);
            }

            this._prevInput.Add(prevInput);


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

            return this._prevOutput.Last();
        }

        protected override NdArray BackwardSingle(NdArray gy)
        {
            var prevInput = this._prevInput.Last()[0];
            var prevOutput = this._prevOutput.Last()[0];

            this._prevInput.RemoveAt(this._prevInput.Count - 1);
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            return this.NeedPreviousBackward(gy, prevInput, prevOutput);
        }

        protected override NdArray[] BackwardSingle(NdArray[] gy)
        {
            var prevInput = this._prevInput.Last();
            var prevOutput = this._prevOutput.Last();

            this._prevInput.RemoveAt(this._prevInput.Count - 1);
            this._prevOutput.RemoveAt(this._prevOutput.Count - 1);

            NdArray[] result = new NdArray[gy.Length];

#if DEBUG
            for (int i = 0; i < gy.Length; i++)
#else
            Parallel.For(0, gy.Length, i =>
#endif
            {
                result[i] = this.NeedPreviousBackward(gy[i], prevInput[i], prevOutput[i]);
            }
#if !DEBUG
            );
#endif

            return result;
        }
    }
}
