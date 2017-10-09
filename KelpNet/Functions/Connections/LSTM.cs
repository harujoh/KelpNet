using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class LSTM : Function
    {
        public Linear upward0;
        public Linear upward1;
        public Linear upward2;
        public Linear upward3;

        public Linear lateral0;
        public Linear lateral1;
        public Linear lateral2;
        public Linear lateral3;

        private List<Real[]> aParam;
        private List<Real[]> iParam;
        private List<Real[]> fParam;
        private List<Real[]> oParam;
        private List<Real[]> cParam;

        private Real[] hParam;

        NdArray gxPrev0;
        NdArray gxPrev1;
        NdArray gxPrev2;
        NdArray gxPrev3;
        Real[] gcPrev;

        private bool initialized = false;

        public LSTM(int inSize, int outSize, Real[,] initialUpwardW = null, Real[] initialUpwardb = null, Real[,] initialLateralW = null, string name = "LSTM") : base(name, inSize, outSize)
        {
            List<FunctionParameter> functionParameters = new List<FunctionParameter>();

            this.upward0 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, gpuEnable: false, name: "upward0");
            this.upward1 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, gpuEnable: false, name: "upward1");
            this.upward2 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, gpuEnable: false, name: "upward2");
            this.upward3 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, gpuEnable: false, name: "upward3");

            functionParameters.AddRange(this.upward0.Parameters);
            functionParameters.AddRange(this.upward1.Parameters);
            functionParameters.AddRange(this.upward2.Parameters);
            functionParameters.AddRange(this.upward3.Parameters);

            //lateralはBiasは無し
            this.lateral0 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, gpuEnable: false, name: "lateral0");
            this.lateral1 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, gpuEnable: false, name: "lateral1");
            this.lateral2 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, gpuEnable: false, name: "lateral2");
            this.lateral3 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, gpuEnable: false, name: "lateral3");

            functionParameters.AddRange(this.lateral0.Parameters);
            functionParameters.AddRange(this.lateral1.Parameters);
            functionParameters.AddRange(this.lateral2.Parameters);
            functionParameters.AddRange(this.lateral3.Parameters);

            this.Parameters = functionParameters.ToArray();

            Forward = ForwardCpu;
            Backward = BackwardCpu;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            Real[][] upwards = new Real[4][];
            upwards[0] = this.upward0.Forward(x).Data;
            upwards[1] = this.upward1.Forward(x).Data;
            upwards[2] = this.upward2.Forward(x).Data;
            upwards[3] = this.upward3.Forward(x).Data;

            if (this.hParam == null)
            {
                //値がなければ初期化
                this.aParam = new List<Real[]>();
                this.iParam = new List<Real[]>();
                this.fParam = new List<Real[]>();
                this.oParam = new List<Real[]>();
                this.cParam = new List<Real[]>();
                this.hParam = new Real[x.BatchCount * this.OutputCount];
                this.gcPrev = new Real[x.BatchCount * this.InputCount];
            }
            else
            {
                NdArray prevInput = new NdArray(this.hParam, new[] { OutputCount }, x.BatchCount);
                Real[] laterals0 = this.lateral0.Forward(prevInput).Data;
                Real[] laterals1 = this.lateral1.Forward(prevInput).Data;
                Real[] laterals2 = this.lateral2.Forward(prevInput).Data;
                Real[] laterals3 = this.lateral3.Forward(prevInput).Data;

                for (int i = 0; i < OutputCount * x.BatchCount; i++)
                {
                    upwards[0][i] += laterals0[i];
                    upwards[1][i] += laterals1[i];
                    upwards[2][i] += laterals2[i];
                    upwards[3][i] += laterals3[i];
                }
            }


            if (this.cParam.Count == 0)
            {
                this.cParam.Add(new Real[this.OutputCount * x.BatchCount]);
            }

            Real[] la = new Real[this.OutputCount * x.BatchCount];
            Real[] li = new Real[this.OutputCount * x.BatchCount];
            Real[] lf = new Real[this.OutputCount * x.BatchCount];
            Real[] lo = new Real[this.OutputCount * x.BatchCount];
            Real[] cPrev = this.cParam[this.cParam.Count - 1];
            Real[] cResult = new Real[cPrev.Length];

            for (int i = 0; i < x.BatchCount; i++)
            {
                //再配置
                for (int j = 0; j < this.OutputCount; j++)
                {
                    int index = j * 4;
                    int batchIndex = j + i * OutputCount;

                    la[batchIndex] = Math.Tanh(upwards[index / this.OutputCount][index % this.OutputCount + i * OutputCount]);
                    li[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + i * OutputCount]);
                    lf[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + i * OutputCount]);
                    lo[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + i * OutputCount]);

                    cResult[batchIndex] = la[batchIndex] * li[batchIndex] + lf[batchIndex] * cPrev[batchIndex];
                    this.hParam[batchIndex] = lo[batchIndex] * Math.Tanh(cResult[batchIndex]);
                }
            }

            //Backward用
            this.cParam.Add(cResult);
            this.aParam.Add(la);
            this.iParam.Add(li);
            this.fParam.Add(lf);
            this.oParam.Add(lo);

            return new NdArray(this.hParam, new[] { OutputCount }, x.BatchCount);
        }

        public NdArray BackwardCpu(NdArray gh)
        {
            BackwardCountUp();

            Real[] lgh = gh.Data.ToArray();

            if (!initialized)
            {
                //値がなければ初期化
                this.gxPrev0 = new NdArray(new[] { OutputCount }, gh.BatchCount);
                this.gxPrev1 = new NdArray(new[] { OutputCount }, gh.BatchCount);
                this.gxPrev2 = new NdArray(new[] { OutputCount }, gh.BatchCount);
                this.gxPrev3 = new NdArray(new[] { OutputCount }, gh.BatchCount);
                initialized = true;
            }
            else
            {
                Real[] ghPre0 = this.lateral0.Backward(this.gxPrev0).Data;
                Real[] ghPre1 = this.lateral1.Backward(this.gxPrev1).Data;
                Real[] ghPre2 = this.lateral2.Backward(this.gxPrev2).Data;
                Real[] ghPre3 = this.lateral3.Backward(this.gxPrev3).Data;

                for (int i = 0; i < gh.BatchCount * OutputCount; i++)
                {
                    lgh[i] += ghPre0[i];
                    lgh[i] += ghPre1[i];
                    lgh[i] += ghPre2[i];
                    lgh[i] += ghPre3[i];
                }
            }

            Real[] lcParam = this.cParam[this.cParam.Count - 1];
            this.cParam.RemoveAt(this.cParam.Count - 1);

            Real[] laParam = this.aParam[this.aParam.Count - 1];
            this.aParam.RemoveAt(this.aParam.Count - 1);

            Real[] liParam = this.iParam[this.iParam.Count - 1];
            this.iParam.RemoveAt(this.iParam.Count - 1);

            Real[] lfParam = this.fParam[this.fParam.Count - 1];
            this.fParam.RemoveAt(this.fParam.Count - 1);

            Real[] loParam = this.oParam[this.oParam.Count - 1];
            this.oParam.RemoveAt(this.oParam.Count - 1);

            Real[] cPrev = this.cParam[this.cParam.Count - 1];

            for (int i = 0; i < gh.BatchCount; i++)
            {
                Real[] gParam = new Real[this.InputCount * 4];

                for (int j = 0; j < this.InputCount; j++)
                {
                    int prevOutputIndex = j + i * this.OutputCount;
                    int prevInputIndex = j + i * this.InputCount;

                    double co = Math.Tanh(lcParam[prevOutputIndex]);

                    this.gcPrev[prevInputIndex] += lgh[prevOutputIndex] * loParam[prevOutputIndex] * GradTanh(co);
                    gParam[j + InputCount * 0] = this.gcPrev[prevInputIndex] * liParam[prevOutputIndex] * GradTanh(laParam[prevOutputIndex]);
                    gParam[j + InputCount * 1] = this.gcPrev[prevInputIndex] * laParam[prevOutputIndex] * GradSigmoid(liParam[prevOutputIndex]);
                    gParam[j + InputCount * 2] = this.gcPrev[prevInputIndex] * cPrev[prevOutputIndex] * GradSigmoid(lfParam[prevOutputIndex]);
                    gParam[j + InputCount * 3] = lgh[prevOutputIndex] * co * GradSigmoid(loParam[prevOutputIndex]);

                    this.gcPrev[prevInputIndex] *= lfParam[prevOutputIndex];
                }

                Real[] resultParam = new Real[this.OutputCount * 4];

                //配置換え
                for (int j = 0; j < this.OutputCount * 4; j++)
                {
                    //暗黙的に切り捨て
                    int index = j / this.OutputCount;
                    resultParam[j % this.OutputCount + index * OutputCount] = gParam[j / 4 + j % 4 * InputCount];
                }

                for (int j = 0; j < OutputCount; j++)
                {
                    this.gxPrev0.Data[i * this.OutputCount + j] = resultParam[j + 0 * this.OutputCount];
                    this.gxPrev1.Data[i * this.OutputCount + j] = resultParam[j + 1 * this.OutputCount];
                    this.gxPrev2.Data[i * this.OutputCount + j] = resultParam[j + 2 * this.OutputCount];
                    this.gxPrev3.Data[i * this.OutputCount + j] = resultParam[j + 3 * this.OutputCount];
                }
            }

            Real[] gArray0 = this.upward0.Backward(this.gxPrev0).Data;
            Real[] gArray1 = this.upward1.Backward(this.gxPrev1).Data;
            Real[] gArray2 = this.upward2.Backward(this.gxPrev2).Data;
            Real[] gArray3 = this.upward3.Backward(this.gxPrev3).Data;

            Real[] gx = new Real[gh.BatchCount * this.InputCount];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = gArray0[i] + gArray1[i] + gArray2[i] + gArray3[i];
            }

            return NdArray.Convert(gx, new[] { InputCount }, gh.BatchCount);
        }

        public override void ResetState()
        {
            initialized = false;
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
