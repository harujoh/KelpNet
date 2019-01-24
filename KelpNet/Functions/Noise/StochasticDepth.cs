using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class StochasticDepth<T> : Function<T> where T : unmanaged, IComparable<T> //SplitFunctionと置き換えるように使用する
    {
        const string FUNCTION_NAME = "StochasticDepth";

        private readonly Real<T> _pl;

        private readonly List<bool> _skipList = new List<bool>();

        private readonly Function<T> _function; //確率でスキップされる
        private readonly Function<T> _resBlock; //必ず実行される

        private bool IsSkip()
        {
            bool result = Mother<T>.Dice.NextDouble() >= this._pl;

            this._skipList.Add(result);

            return result;
        }

        public StochasticDepth(Function<T> function, Function<T> resBlock = null, double pl = 0.5f, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._function = function;
            this._resBlock = resBlock;

            this._pl = pl;
        }

        public override NdArray<T>[] Forward(params NdArray<T>[] xs)
        {
            List<NdArray<T>> resultArray = new List<NdArray<T>>();
            NdArray<T>[] resResult = xs;

            if (_resBlock != null)
            {
                resResult = _resBlock.Forward(xs);
            }

            resultArray.AddRange(resResult);

            if (!IsSkip())
            {
                Real<T> scale = 1 / (1 - this._pl);
                NdArray<T>[] result = _function.Forward(xs);

                for (int i = 0; i < result.Length; i++)
                {
                    for (int j = 0; j < result[i].Data.Length; j++)
                    {
                        result[i].Data[j] *= scale;
                    }
                }

                resultArray.AddRange(result);
            }
            else
            {
                NdArray<T>[] result = new NdArray<T>[resResult.Length];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray<T>(resResult[i].Shape, resResult[i].BatchCount, resResult[i].ParentFunc);
                }

                resultArray.AddRange(result);
            }

            return resultArray.ToArray();
        }

        public override void Backward(params NdArray<T>[] ys)
        {
            if (_resBlock != null)
            {
                _resBlock.Backward(ys);
            }

            bool isSkip = this._skipList[this._skipList.Count - 1];
            this._skipList.RemoveAt(this._skipList.Count - 1);

            if (!isSkip)
            {
                NdArray<T>[] copyys = new NdArray<T>[ys.Length];

                Real<T> scale = 1 / (1 - this._pl);

                for (int i = 0; i < ys.Length; i++)
                {
                    copyys[i] = ys[i].Clone();

                    for (int j = 0; j < ys[i].Data.Length; j++)
                    {
                        copyys[i].Data[j] *= scale;
                    }
                }

                _function.Backward(copyys);
            }
        }

        public override NdArray<T>[] Predict(params NdArray<T>[] xs)
        {
            return _function.Predict(xs);
        }
    }
}
