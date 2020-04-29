using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    [Serializable]
    public class StochasticDepth<T> : Function<T> where T : unmanaged, IComparable<T> //SplitFunctionと置き換えるように使用する
    {
        const string FUNCTION_NAME = "StochasticDepth";

        private T _pl;

        private readonly List<bool> _skipList = new List<bool>();

        private readonly Function<T> _function; //確率でスキップされる
        private readonly Function<T> _resBlock; //必ず実行される

        public StochasticDepth(Function<T> function, Function<T> resBlock = null, double pl = 0.5, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._function = function;
            this._resBlock = resBlock;

            switch (this)
            {
                case StochasticDepth<float> stochasticDepthF:
                    stochasticDepthF._pl = (float)pl;
                    break;

                case StochasticDepth<double> stochasticDepthD:
                    stochasticDepthD._pl = pl;
                    break;
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case StochasticDepth<float> stochasticDepthF:
                    stochasticDepthF.Forward = (x) => StochasticDepthF.Forward(stochasticDepthF._pl, stochasticDepthF._skipList, stochasticDepthF._function, stochasticDepthF._resBlock, x);
                    stochasticDepthF.Backward = (ys) => StochasticDepthF.Backward(stochasticDepthF._pl, stochasticDepthF._skipList, stochasticDepthF._function, stochasticDepthF._resBlock, ys);
                    break;
                case StochasticDepth<double> stochasticDepthD:
                    stochasticDepthD.Forward = (x) => StochasticDepthD.Forward(stochasticDepthD._pl, stochasticDepthD._skipList, stochasticDepthD._function, stochasticDepthD._resBlock, x);
                    stochasticDepthD.Backward = (ys)=>StochasticDepthD.Backward(stochasticDepthD._pl, stochasticDepthD._skipList, stochasticDepthD._function, stochasticDepthD._resBlock, ys);
                    break;
            }

            this.Predict = (xs) =>_function.Predict(xs);
        }
    }
#endif

#if DOUBLE
    public static class StochasticDepthD
#else
    public static class StochasticDepthF
#endif
    {
        static bool IsSkip(Real pl, List<bool> skipList)
        {
            bool result = (Real)Mother.Dice.NextDouble() >= pl;

            skipList.Add(result);

            return result;
        }

        public static NdArray<Real>[] Forward(Real pl, List<bool> skipList, Function<Real> function, Function<Real> resBlock, params NdArray<Real>[] xs)
        {
            List<NdArray<Real>> resultArray = new List<NdArray<Real>>();
            NdArray<Real>[] resResult = xs;

            if (resBlock != null)
            {
                resResult = resBlock.Forward(xs);
            }

            resultArray.AddRange(resResult);

            if (!IsSkip(pl, skipList))
            {
                Real scale = 1 / (1 - pl);
                NdArray<Real>[] result = function.Forward(xs);

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
                NdArray<Real>[] result = new NdArray<Real>[resResult.Length];

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new NdArray<Real>(resResult[i].Shape, resResult[i].BatchCount, resResult[i].ParentFunc);
                }

                resultArray.AddRange(result);
            }

            return resultArray.ToArray();
        }

        public static void Backward(Real pl, List<bool> skipList, Function<Real> function, Function<Real> resBlock, params NdArray<Real>[] ys)
        {
            if (resBlock != null)
            {
                resBlock.Backward(ys);
            }

            bool isSkip = skipList[skipList.Count - 1];
            skipList.RemoveAt(skipList.Count - 1);

            if (!isSkip)
            {
                NdArray<Real>[] copyys = new NdArray<Real>[ys.Length];

                Real scale = 1 / (1 - pl);

                for (int i = 0; i < ys.Length; i++)
                {
                    copyys[i] = ys[i].Clone();

                    for (int j = 0; j < ys[i].Data.Length; j++)
                    {
                        copyys[i].Data[j] *= scale;
                    }
                }

                function.Backward(copyys);
            }
        }
    }
}
