using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using KelpNet.CPU;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
#if !DOUBLE
    [Serializable]
    public class LSTM<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "LSTM";

        public Linear<T> upward;

        public Linear<T> lateral;

        private List<T[][]> paramLists = new List<T[][]>(); //0:cPrev 1:a 2:i 3:f 4:o 5:c
        private List<T[][]> usedParamLists = new List<T[][]>();

        private List<NdArray<T>> hPrevParams = new List<NdArray<T>>();
        private List<NdArray<T>> hUsedPrevParams = new List<NdArray<T>>();

        private List<T[]> gxPrevGrads = new List<T[]>();

        public NdArray<T> hParam;
        private T[] cPrev = { };

        public readonly int OutputCount;

        public LSTM(int inSize, int outSize, Array lateralInit = null, Array upwardInit = null, Array biasInit = null, Array forgetBiasInit = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.OutputCount = outSize;

            List<NdArray<T>> functionParameters = new List<NdArray<T>>();

            T[] lateralW = null;
            T[] upwardW = null;
            T[] upwardb = null;

            if (upwardInit != null)
            {
                upwardW = new T[inSize * outSize * 4];

                for (int i = 0; i < 4; i++)
                {
                    Buffer.BlockCopy(upwardInit, 0, upwardW, i * upwardInit.Length * Marshal.SizeOf<T>(), upwardInit.Length * Marshal.SizeOf<T>());
                }
            }

            if (lateralInit != null)
            {
                lateralW = new T[outSize * outSize * 4];

                for (int i = 0; i < 4; i++)
                {
                    Buffer.BlockCopy(lateralInit, 0, lateralW, i * lateralInit.Length * Marshal.SizeOf<T>(), lateralInit.Length * Marshal.SizeOf<T>());
                }
            }

            if (biasInit != null && forgetBiasInit != null)
            {
                upwardb = new T[outSize * 4];

                T[] tmpBiasInit = biasInit.FlattenEx<T>();

                for (int i = 0; i < biasInit.Length; i++)
                {
                    upwardb[i * 4 + 0] = tmpBiasInit[i];
                    upwardb[i * 4 + 1] = tmpBiasInit[i];
                    upwardb[i * 4 + 3] = tmpBiasInit[i];
                }

                T[] tmpforgetBiasInit = forgetBiasInit.FlattenEx<T>();

                for (int i = 0; i < tmpforgetBiasInit.Length; i++)
                {
                    upwardb[i * 4 + 2] = tmpforgetBiasInit[i];
                }
            }

            this.upward = new Linear<T>(inSize, outSize * 4, noBias: false, initialW: upwardW, initialb: upwardb, name: "upward");
            functionParameters.AddRange(this.upward.Parameters);

            //lateralはBiasは無し
            this.lateral = new Linear<T>(outSize, outSize * 4, noBias: true, initialW: lateralW, name: "lateral");
            functionParameters.AddRange(this.lateral.Parameters);

            this.Parameters = functionParameters.ToArray();

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case LSTM<float> lstmF:
                    lstmF.SingleInputForward = (x) => LSTMF.SingleInputForward(x, lstmF.upward, lstmF.lateral, lstmF.paramLists, lstmF.hPrevParams, ref lstmF.hParam, ref lstmF.cPrev, lstmF.OutputCount, lstmF);
                    lstmF.SingleOutputBackward = (y, x) => LSTMF.SingleOutputBackward(y, lstmF.upward, lstmF.lateral, lstmF.paramLists, lstmF.hPrevParams, lstmF.usedParamLists, lstmF.hUsedPrevParams, lstmF.gxPrevGrads, lstmF.OutputCount, lstmF.Backward);
                    break;

                case LSTM<double> lstmD:
                    lstmD.SingleInputForward = (x) => LSTMD.SingleInputForward(x, lstmD.upward, lstmD.lateral, lstmD.paramLists, lstmD.hPrevParams, ref lstmD.hParam, ref lstmD.cPrev, lstmD.OutputCount, lstmD);
                    lstmD.SingleOutputBackward = (y, x) => LSTMD.SingleOutputBackward(y, lstmD.upward, lstmD.lateral, lstmD.paramLists, lstmD.hPrevParams, lstmD.usedParamLists, lstmD.hUsedPrevParams, lstmD.gxPrevGrads, lstmD.OutputCount, lstmD.Backward);
                    break;
            }
        }

        public override void ResetState()
        {
            base.ResetState();
            this.hParam = null;

            this.paramLists = new List<T[][]>();
            this.hPrevParams = new List<NdArray<T>>();

            this.usedParamLists = new List<T[][]>();
            this.hUsedPrevParams = new List<NdArray<T>>();

            this.gxPrevGrads = new List<T[]>();
        }
    }
#endif

#if DOUBLE
    public static class LSTMD
#else
    public static class LSTMF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> upward, IFunction<Real> lateral, List<Real[][]> paramList, List<NdArray<Real>> hPrevParams, ref NdArray<Real> hParam, ref Real[] lcPrev, int outputCount, IFunction<Real> lstm)
        {
            int outputDataSize = x.BatchCount * outputCount;

            NdArray<Real> lstmIn = upward.Forward(x)[0];

            if (hParam == null)
            {
                lcPrev = new Real[outputDataSize];
            }
            else
            {
                NdArray<Real> hPrevParam = hParam.Clone();
                if (hPrevParam.Grad != null) hPrevParam.InitGrad();
                lstmIn += lateral.Forward(hPrevParam)[0];
                hPrevParams.Add(hPrevParam);
            }

            //0:cPrev 1:a 2:i 3:f 4:o 5:c
            Real[][] param = { lcPrev, new Real[outputDataSize], new Real[outputDataSize], new Real[outputDataSize], new Real[outputDataSize], new Real[outputDataSize] };
            Real[] lhParam = new Real[outputDataSize];

            int index = 0;
            for (int outIndex = 0; outIndex < lhParam.Length; outIndex++)
            {
                param[1][outIndex] = Math.Tanh(lstmIn.Data[index++]);
                param[2][outIndex] = Sigmoid(lstmIn.Data[index++]);
                param[3][outIndex] = Sigmoid(lstmIn.Data[index++]);
                param[4][outIndex] = Sigmoid(lstmIn.Data[index++]);

                param[5][outIndex] = param[1][outIndex] * param[2][outIndex] + param[3][outIndex] * param[0][outIndex];

                lhParam[outIndex] = param[4][outIndex] * Math.Tanh(param[5][outIndex]);
            }

            paramList.Add(param);

            //Backwardで消えないように別で保管
            lcPrev = param[5];

            hParam = new NdArray<Real>(lhParam, new[] { outputCount }, x.BatchCount, lstm);
            return hParam;
        }

        public static void SingleOutputBackward(NdArray<Real> y, IFunction<Real> upward, IFunction<Real> lateral, List<Real[][]> paramLists, List<NdArray<Real>> hPrevParams, List<Real[][]> usedParamLists, List<NdArray<Real>> hUsedPrevParams, List<Real[]> gxPrevGrads, int outputCount, ActionOptional<Real> backward)
        {
            Real[] gxPrevGrad = new Real[y.BatchCount * outputCount * 4];
            Real[] gcPrev = new Real[y.BatchCount * outputCount];

            //0:cPrev 1:a 2:i 3:f 4:o 5:c
            Real[][] param = paramLists[paramLists.Count - 1];
            paramLists.RemoveAt(paramLists.Count - 1);
            usedParamLists.Add(param);

            int index = 0;
            for (int prevOutputIndex = 0; prevOutputIndex < gcPrev.Length; prevOutputIndex++)
            {
                Real co = Math.Tanh(param[5][prevOutputIndex]);

                gcPrev[prevOutputIndex] += y.Grad[prevOutputIndex] * param[4][prevOutputIndex] * GradTanh(co);

                gxPrevGrad[index++] = gcPrev[prevOutputIndex] * param[2][prevOutputIndex] * GradTanh(param[1][prevOutputIndex]);
                gxPrevGrad[index++] = gcPrev[prevOutputIndex] * param[1][prevOutputIndex] * GradSigmoid(param[2][prevOutputIndex]);
                gxPrevGrad[index++] = gcPrev[prevOutputIndex] * param[0][prevOutputIndex] * GradSigmoid(param[3][prevOutputIndex]);
                gxPrevGrad[index++] = y.Grad[prevOutputIndex] * co * GradSigmoid(param[4][prevOutputIndex]);

                gcPrev[prevOutputIndex] *= param[3][prevOutputIndex];
            }

            gxPrevGrads.Add(gxPrevGrad);

            if (hPrevParams.Count > 0)
            {
                //linearのBackwardはgxPrev.Gradしか使わないのでgxPrev.Dataは空
                NdArray<Real> gxPrev = new NdArray<Real>(new[] { outputCount * 4 }, y.BatchCount);
                gxPrev.Grad = gxPrevGrad;
                lateral.Backward(gxPrev);

                NdArray<Real> hPrevParam = hPrevParams[hPrevParams.Count - 1];
                hPrevParams.RemoveAt(hPrevParams.Count - 1);
                hUsedPrevParams.Add(hPrevParam);

                //hのBakckward
                backward(hPrevParam);

                //使い切ったら戻す
                if (hPrevParams.Count == 0)
                {
                    hPrevParams.AddRange(hUsedPrevParams);
                    hUsedPrevParams.Clear();
                }
            }

            //linearのBackwardはgy.Gradしか使わないのでgy.Dataは空
            NdArray<Real> gy = new NdArray<Real>(new[] { outputCount * 4 }, y.BatchCount);
            gy.Grad = gxPrevGrads[0];
            gxPrevGrads.RemoveAt(0);
            upward.Backward(gy);

            //使い切ったら戻す
            if (paramLists.Count == 0)
            {
                paramLists.AddRange(usedParamLists);
                usedParamLists.Clear();
            }
        }


        static Real Sigmoid(Real x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        static Real GradSigmoid(Real x)
        {
            return x * (1 - x);
        }

        static Real GradTanh(Real x)
        {
            return 1 - x * x;
        }
    }
}
