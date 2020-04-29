using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using KelpNet.CPU;
#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
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

        private List<T[]> aParam = new List<T[]>();
        private List<T[]> iParam = new List<T[]>();
        private List<T[]> fParam = new List<T[]>();
        private List<T[]> oParam = new List<T[]>();
        public List<T[]> cNextParam = new List<T[]>();
        private List<T[]> cPrevParam = new List<T[]>();
        private List<NdArray<T>> hPrevParams = new List<NdArray<T>>();

        private List<T[]> aUsedParam = new List<T[]>();
        private List<T[]> iUsedParam = new List<T[]>();
        private List<T[]> fUsedParam =new List<T[]>();
        private List<T[]> oUsedParam = new List<T[]>();
        public List<T[]> cUsedNextParam = new List<T[]>();
        private List<T[]> cUsedPrevParam = new List<T[]>();
        private List<NdArray<T>> hUsedPrevParams = new List<NdArray<T>>();

        private List<T[]> gxPrevGrads = new List<T[]>();

        public NdArray<T> hParam;
        private T[] cPrev = {};

        public readonly int InputCount;
        public readonly int OutputCount;

        public LSTM(int inSize, int outSize, Array lateralInit = null, Array upwardInit = null, Array biasInit = null, Array forgetBiasInit = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.InputCount = inSize;
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
                    lstmF.SingleInputForward = (x) => LSTMF.SingleInputForward(x, lstmF.upward, lstmF.lateral, lstmF.aParam, lstmF.iParam, lstmF.fParam, lstmF.oParam, lstmF.cNextParam, lstmF.cPrevParam, lstmF.hPrevParams, ref lstmF.hParam,ref lstmF.cPrev, lstmF.OutputCount, lstmF);
                    lstmF.SingleOutputBackward = (y, x) => LSTMF.SingleOutputBackward(y, x, lstmF.upward, lstmF.lateral, lstmF.aParam, lstmF.iParam, lstmF.fParam, lstmF.oParam, lstmF.cNextParam, lstmF.cPrevParam, lstmF.hPrevParams, lstmF.aUsedParam, lstmF.iUsedParam, lstmF.fUsedParam, lstmF.oUsedParam, lstmF.cUsedNextParam, lstmF.cUsedPrevParam, lstmF.hUsedPrevParams, lstmF.gxPrevGrads, lstmF.OutputCount, lstmF.Backward);
                    break;

                case LSTM<double> lstmD:
                    lstmD.SingleInputForward = (x) => LSTMD.SingleInputForward(x, lstmD.upward, lstmD.lateral, lstmD.aParam, lstmD.iParam, lstmD.fParam, lstmD.oParam, lstmD.cNextParam, lstmD.cPrevParam, lstmD.hPrevParams, ref lstmD.hParam,ref lstmD.cPrev, lstmD.OutputCount, lstmD);
                    lstmD.SingleOutputBackward = (y, x) => LSTMD.SingleOutputBackward(y, x, lstmD.upward, lstmD.lateral, lstmD.aParam, lstmD.iParam, lstmD.fParam, lstmD.oParam, lstmD.cNextParam, lstmD.cPrevParam, lstmD.hPrevParams, lstmD.aUsedParam, lstmD.iUsedParam, lstmD.fUsedParam, lstmD.oUsedParam, lstmD.cUsedNextParam, lstmD.cUsedPrevParam, lstmD.hUsedPrevParams, lstmD.gxPrevGrads, lstmD.OutputCount, lstmD.Backward);
                    break;
            }
        }

        public override void ResetState()
        {
            base.ResetState();
            this.hParam = null;

            this.aParam = new List<T[]>();
            this.iParam = new List<T[]>();
            this.fParam = new List<T[]>();
            this.oParam = new List<T[]>();
            this.cNextParam = new List<T[]>();
            this.cPrevParam = new List<T[]>();
            this.hPrevParams = new List<NdArray<T>>();

            this.aUsedParam = new List<T[]>();
            this.iUsedParam = new List<T[]>();
            this.fUsedParam = new List<T[]>();
            this.oUsedParam = new List<T[]>();
            this.cUsedNextParam = new List<T[]>();
            this.cUsedPrevParam = new List<T[]>();
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
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, IFunction<Real> upward, IFunction<Real> lateral, List<Real[]> aParam, List<Real[]> iParam, List<Real[]> fParam, List<Real[]> oParam, List<Real[]> cNextParam, List<Real[]> cPrevParam, List<NdArray<Real>> hPrevParams, ref NdArray<Real> hParam,ref Real[] cPrev, int OutputCount, IFunction<Real> lstm)
        {
            NdArray<Real> lstmIn = upward.Forward(x)[0];

            int outputDataSize = x.BatchCount * OutputCount;

            if (hParam == null)
            {
                cPrev = new Real[outputDataSize];
            }
            else
            {
                NdArray<Real> hPrevParam = hParam.Clone();
                if (hPrevParam.Grad != null) hPrevParam.InitGrad();
                lstmIn += lateral.Forward(hPrevParam)[0];
                hPrevParams.Add(hPrevParam);
            }

            Real[] la = new Real[outputDataSize];
            Real[] li = new Real[outputDataSize];
            Real[] lf = new Real[outputDataSize];
            Real[] lo = new Real[outputDataSize];
            Real[] cNext = new Real[outputDataSize];
            Real[] lhParam = new Real[outputDataSize];

            for (int b = 0; b < x.BatchCount; b++)
            {
                int index = b * lstmIn.Length;

                for (int i = 0; i < OutputCount; i++)
                {
                    int outIndex = b * OutputCount + i;

                    la[outIndex] = KelpMath.Tanh(lstmIn.Data[index++]);
                    li[outIndex] = Sigmoid(lstmIn.Data[index++]);
                    lf[outIndex] = Sigmoid(lstmIn.Data[index++]);
                    lo[outIndex] = Sigmoid(lstmIn.Data[index++]);

                    cNext[outIndex] = la[outIndex] * li[outIndex] + lf[outIndex] * cPrev[outIndex];

                    lhParam[outIndex] = lo[outIndex] * KelpMath.Tanh(cNext[outIndex]);
                }
            }

            cPrevParam.Add(cPrev);
            cNextParam.Add(cNext);
            aParam.Add(la);
            iParam.Add(li);
            fParam.Add(lf);
            oParam.Add(lo);

            //Backwardで消えないように別で保管
            cPrev = cNext;

            hParam = new NdArray<Real>(lhParam, new[] { OutputCount }, x.BatchCount, lstm);
            return hParam;
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, IFunction<Real> upward, IFunction<Real> lateral, List<Real[]> aParam, List<Real[]> iParam, List<Real[]> fParam, List<Real[]> oParam, List<Real[]> cNextParam, List<Real[]> cPrevParam, List<NdArray<Real>> hPrevParams, List<Real[]> aUsedParam, List<Real[]> iUsedParam, List<Real[]> fUsedParam, List<Real[]> oUsedParam, List<Real[]> cUsedNextParam, List<Real[]> cUsedPrevParam, List<NdArray<Real>> hUsedPrevParams, List<Real[]> gxPrevGrads, int OutputCount, ActionOptional<Real> Backward)
        {
            Real[] gxPrevGrad = new Real[y.BatchCount * OutputCount * 4];

            Real[] gcPrev = new Real[y.BatchCount * OutputCount];

            Real[] lcNextParam = cNextParam[cPrevParam.Count - 1];
            cNextParam.RemoveAt(cNextParam.Count - 1);
            cUsedNextParam.Add(lcNextParam);

            Real[] tanh_a = aParam[aParam.Count - 1];
            aParam.RemoveAt(aParam.Count - 1);
            aUsedParam.Add(tanh_a);

            Real[] sig_i = iParam[iParam.Count - 1];
            iParam.RemoveAt(iParam.Count - 1);
            iUsedParam.Add(sig_i);

            Real[] sig_f = fParam[fParam.Count - 1];
            fParam.RemoveAt(fParam.Count - 1);
            fUsedParam.Add(sig_f);

            Real[] sig_o = oParam[oParam.Count - 1];
            oParam.RemoveAt(oParam.Count - 1);
            oUsedParam.Add(sig_o);

            Real[] lcPrev = cPrevParam[cPrevParam.Count - 1];
            cPrevParam.RemoveAt(cPrevParam.Count - 1);
            cUsedPrevParam.Add(lcPrev);

            for (int b = 0; b < y.BatchCount; b++)
            {
                int index = b * OutputCount * 4;

                for (int i = 0; i < OutputCount; i++)
                {
                    int prevOutputIndex = b * OutputCount + i;

                    Real co = KelpMath.Tanh(lcNextParam[prevOutputIndex]);

                    gcPrev[prevOutputIndex] += y.Grad[prevOutputIndex] * sig_o[prevOutputIndex] * GradTanh(co);

                    gxPrevGrad[index++] = gcPrev[prevOutputIndex] * sig_i[prevOutputIndex] * GradTanh(tanh_a[prevOutputIndex]);
                    gxPrevGrad[index++] = gcPrev[prevOutputIndex] * tanh_a[prevOutputIndex] * GradSigmoid(sig_i[prevOutputIndex]);
                    gxPrevGrad[index++] = gcPrev[prevOutputIndex] * lcPrev[prevOutputIndex] * GradSigmoid(sig_f[prevOutputIndex]);
                    gxPrevGrad[index++] = y.Grad[prevOutputIndex] * co * GradSigmoid(sig_o[prevOutputIndex]);

                    gcPrev[prevOutputIndex] *= sig_f[prevOutputIndex];
                }
            }

            gxPrevGrads.Add(gxPrevGrad);

            //gxPrevをlateralとupwardに渡すことでaddのBackwardを兼ねる
            if (hPrevParams.Count > 0)
            {
                NdArray<Real> gxPrev = new NdArray<Real>(new[] { OutputCount * 4 }, y.BatchCount);
                gxPrev.Grad = gxPrevGrad;
                lateral.Backward(gxPrev);

                NdArray<Real> hPrevParam = hPrevParams[hPrevParams.Count - 1];
                hPrevParams.RemoveAt(hPrevParams.Count - 1);
                hUsedPrevParams.Add(hPrevParam);

                //hのBakckward
                Backward(hPrevParam);

                //使い切ったら戻す
                if (hPrevParams.Count == 0)
                {
                    hPrevParams.AddRange(hUsedPrevParams);
                    hUsedPrevParams.Clear();
                }
            }

            NdArray<Real> gy = new NdArray<Real>(new[] { OutputCount * 4 }, y.BatchCount);
            gy.Grad = gxPrevGrads[0];
            gxPrevGrads.RemoveAt(0);
            upward.Backward(gy);

            //linearのBackwardはgxPrev.Gradしか使わないのでgxPrev.Dataは空
            //使い切ったら戻す
            if (cNextParam.Count == 0)
            {
                cNextParam.AddRange(cUsedNextParam);
                aParam.AddRange(aUsedParam);
                iParam.AddRange(iUsedParam);
                fParam.AddRange(fUsedParam);
                oParam.AddRange(oUsedParam);
                cPrevParam.AddRange(cUsedPrevParam);
                cUsedNextParam.Clear();
                aUsedParam.Clear();
                iUsedParam.Clear();
                fUsedParam.Clear();
                oUsedParam.Clear();
                cUsedPrevParam.Clear();
            }
        }


        static Real Sigmoid(Real x)
        {
            return 1 / (1 + KelpMath.Exp(-x));
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
