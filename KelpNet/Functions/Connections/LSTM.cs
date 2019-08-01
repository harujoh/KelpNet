using System;
using System.Collections.Generic;
using KelpNet.CPU;

namespace KelpNet
{
    [Serializable]
    public class LSTM : SingleInputFunction
    {
        const string FUNCTION_NAME = "LSTM";

        public Linear upward;

        public Linear lateral;

        private List<Real[]> aParam;
        private List<Real[]> iParam;
        private List<Real[]> fParam;
        private List<Real[]> oParam;
        public List<Real[]> cNextParam;
        private List<Real[]> cPrevParam;
        private List<NdArray> hPrevParams;

        private List<Real[]> aUsedParam;
        private List<Real[]> iUsedParam;
        private List<Real[]> fUsedParam;
        private List<Real[]> oUsedParam;
        public List<Real[]> cUsedNextParam;
        private List<Real[]> cUsedPrevParam;
        private List<NdArray> hUsedPrevParams;

        private List<Real[]> gxPrevGrads;

        public NdArray hParam;
        private Real[] cPrev;

        public readonly int InputCount;
        public readonly int OutputCount;

        public LSTM(int inSize, int outSize, Array lateralInit = null, Array upwardInit = null, Array biasInit = null, Array forgetBiasInit = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.InputCount = inSize;
            this.OutputCount = outSize;

            List<NdArray> functionParameters = new List<NdArray>();

            Real[] lateralW = null;
            Real[] upwardW = null;
            Real[] upwardb = null;

            if (upwardInit != null)
            {
                upwardW = new Real[inSize * outSize * 4];

                Real[] tmpUpwardInit = Real.ToRealArray(upwardInit);

                for (int i = 0; i < 4; i++)
                {
                    Array.Copy(tmpUpwardInit, 0, upwardW, i * tmpUpwardInit.Length, tmpUpwardInit.Length);
                }
            }

            if (lateralInit != null)
            {
                lateralW = new Real[outSize * outSize * 4];

                Real[] tmpLateralInit = Real.ToRealArray(lateralInit);

                for (int i = 0; i < 4; i++)
                {
                    Array.Copy(tmpLateralInit, 0, lateralW, i * tmpLateralInit.Length, tmpLateralInit.Length);
                }
            }

            if (biasInit != null && forgetBiasInit != null)
            {
                upwardb = new Real[outSize * 4];

                Real[] tmpBiasInit = Real.ToRealArray(biasInit);

                for (int i = 0; i < biasInit.Length; i++)
                {
                    upwardb[i * 4 + 0] = tmpBiasInit[i];
                    upwardb[i * 4 + 1] = tmpBiasInit[i];
                    upwardb[i * 4 + 3] = tmpBiasInit[i];
                }

                Real[] tmpforgetBiasInit = Real.ToRealArray(forgetBiasInit);

                for (int i = 0; i < tmpforgetBiasInit.Length; i++)
                {
                    upwardb[i * 4 + 2] = tmpforgetBiasInit[i];
                }
            }

            this.upward = new Linear(inSize, outSize * 4, noBias: false, initialW: upwardW, initialb: upwardb, name: "upward");
            functionParameters.AddRange(this.upward.Parameters);

            //lateralはBiasは無し
            this.lateral = new Linear(outSize, outSize * 4, noBias: true, initialW: lateralW, name: "lateral");
            functionParameters.AddRange(this.lateral.Parameters);

            this.Parameters = functionParameters.ToArray();
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            NdArray lstmIn = this.upward.Forward(x)[0]; //a

            int outputDataSize = x.BatchCount * this.OutputCount;

            if (this.hParam == null)
            {
                //値がなければ初期化
                this.aParam = new List<Real[]>();
                this.iParam = new List<Real[]>();
                this.fParam = new List<Real[]>();
                this.oParam = new List<Real[]>();
                this.cNextParam = new List<Real[]>();
                this.cPrevParam = new List<Real[]>();
                this.hPrevParams = new List<NdArray>();

                this.aUsedParam = new List<Real[]>();
                this.iUsedParam = new List<Real[]>();
                this.fUsedParam = new List<Real[]>();
                this.oUsedParam = new List<Real[]>();
                this.cUsedNextParam = new List<Real[]>();
                this.cUsedPrevParam = new List<Real[]>();
                this.hUsedPrevParams = new List<NdArray>();

                gxPrevGrads = new List<Real[]>();

                cPrev = new Real[outputDataSize];
            }
            else
            {
                NdArray hPrevParam = this.hParam.Clone();
                if (hPrevParam.Grad != null) hPrevParam.InitGrad();
                lstmIn += this.lateral.Forward(hPrevParam)[0];
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

                for (int i = 0; i < this.OutputCount; i++)
                {
                    int outIndex = b * this.OutputCount + i;

                    la[outIndex] = Math.Tanh(lstmIn.Data[index++]);
                    li[outIndex] = Sigmoid(lstmIn.Data[index++]);
                    lf[outIndex] = Sigmoid(lstmIn.Data[index++]);
                    lo[outIndex] = Sigmoid(lstmIn.Data[index++]);

                    cNext[outIndex] = la[outIndex] * li[outIndex] + lf[outIndex] * cPrev[outIndex];

                    lhParam[outIndex] = lo[outIndex] * Math.Tanh(cNext[outIndex]);
                }
            }

            this.cPrevParam.Add(cPrev);
            this.cNextParam.Add(cNext);
            this.aParam.Add(la);
            this.iParam.Add(li);
            this.fParam.Add(lf);
            this.oParam.Add(lo);

            //Backwardで消えないように別で保管
            cPrev = cNext;

            this.hParam = new NdArray(lhParam, new[] { OutputCount }, x.BatchCount, this);
            return this.hParam;
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            Real[] gxPrevGrad = new Real[y.BatchCount * OutputCount * 4];

            Real[] gcPrev = new Real[y.BatchCount * this.OutputCount];

            Real[] lcNextParam = this.cNextParam[this.cPrevParam.Count - 1];
            this.cNextParam.RemoveAt(this.cNextParam.Count - 1);
            this.cUsedNextParam.Add(lcNextParam);

            Real[] tanh_a = this.aParam[this.aParam.Count - 1];
            this.aParam.RemoveAt(this.aParam.Count - 1);
            this.aUsedParam.Add(tanh_a);

            Real[] sig_i = this.iParam[this.iParam.Count - 1];
            this.iParam.RemoveAt(this.iParam.Count - 1);
            this.iUsedParam.Add(sig_i);

            Real[] sig_f = this.fParam[this.fParam.Count - 1];
            this.fParam.RemoveAt(this.fParam.Count - 1);
            this.fUsedParam.Add(sig_f);

            Real[] sig_o = this.oParam[this.oParam.Count - 1];
            this.oParam.RemoveAt(this.oParam.Count - 1);
            this.oUsedParam.Add(sig_o);

            Real[] lcPrev = this.cPrevParam[this.cPrevParam.Count - 1];
            this.cPrevParam.RemoveAt(this.cPrevParam.Count - 1);
            this.cUsedPrevParam.Add(lcPrev);

            for (int b = 0; b < y.BatchCount; b++)
            {
                int index = b * OutputCount * 4;

                for (int i = 0; i < this.OutputCount; i++)
                {
                    int prevOutputIndex = b * this.OutputCount + i;

                    double co = Math.Tanh(lcNextParam[prevOutputIndex]);

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
                NdArray gxPrev = new NdArray(new[] { OutputCount * 4 }, y.BatchCount);
                gxPrev.Grad = gxPrevGrad;
                this.lateral.Backward(gxPrev);

                NdArray hPrevParam = hPrevParams[hPrevParams.Count - 1];
                hPrevParams.RemoveAt(hPrevParams.Count - 1);
                hUsedPrevParams.Add(hPrevParam);

                //hのBakckward
                this.Backward(hPrevParam);

                //使い切ったら戻す
                if (hPrevParams.Count == 0)
                {
                    hPrevParams.AddRange(hUsedPrevParams);
                    hUsedPrevParams.Clear();
                }
            }

            NdArray gy = new NdArray(new[] { OutputCount * 4 }, y.BatchCount);
            gy.Grad = gxPrevGrads[0];
            gxPrevGrads.RemoveAt(0);
            this.upward.Backward(gy);

            //linearのBackwardはgxPrev.Gradしか使わないのでgxPrev.Dataは空
            //使い切ったら戻す
            if (cNextParam.Count == 0)
            {
                this.cNextParam.AddRange(cUsedNextParam);
                this.aParam.AddRange(aUsedParam);
                this.iParam.AddRange(iUsedParam);
                this.fParam.AddRange(fUsedParam);
                this.oParam.AddRange(oUsedParam);
                this.cPrevParam.AddRange(cUsedPrevParam);
                this.cUsedNextParam.Clear();
                this.aUsedParam.Clear();
                this.iUsedParam.Clear();
                this.fUsedParam.Clear();
                this.oUsedParam.Clear();
                this.cUsedPrevParam.Clear();
            }
        }

        public override void ResetState()
        {
            base.ResetState();
            this.hParam = null;
        }

        static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        static double GradSigmoid(double x)
        {
            return x * (1 - x);
        }

        static double GradTanh(double x)
        {
            return 1 - x * x;
        }
    }
}
