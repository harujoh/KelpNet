using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class LSTM : SingleInputFunction
    {
        const string FUNCTION_NAME = "LSTM";

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

        private NdArray hParam;

        NdArray gxPrev0;
        NdArray gxPrev1;
        NdArray gxPrev2;
        NdArray gxPrev3;
        Real[] gcPrev;

        public readonly int InputCount;
        public readonly int OutputCount;

        public LSTM(int inSize, int outSize, Real[,] initialUpwardW = null, Real[] initialUpwardb = null, Real[,] initialLateralW = null, string name = FUNCTION_NAME, bool gpuEnable = false) : base(name)
        {
            this.InputCount = inSize;
            this.OutputCount = outSize;

            List<NdArray> functionParameters = new List<NdArray>();

            this.upward0 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward0", gpuEnable: gpuEnable);
            this.upward1 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward1", gpuEnable: gpuEnable);
            this.upward2 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward2", gpuEnable: gpuEnable);
            this.upward3 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward3", gpuEnable: gpuEnable);

            functionParameters.AddRange(this.upward0.Parameters);
            functionParameters.AddRange(this.upward1.Parameters);
            functionParameters.AddRange(this.upward2.Parameters);
            functionParameters.AddRange(this.upward3.Parameters);

            //lateralはBiasは無し
            this.lateral0 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral0", gpuEnable: gpuEnable);
            this.lateral1 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral1", gpuEnable: gpuEnable);
            this.lateral2 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral2", gpuEnable: gpuEnable);
            this.lateral3 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral3", gpuEnable: gpuEnable);

            functionParameters.AddRange(this.lateral0.Parameters);
            functionParameters.AddRange(this.lateral1.Parameters);
            functionParameters.AddRange(this.lateral2.Parameters);
            functionParameters.AddRange(this.lateral3.Parameters);

            this.Parameters = functionParameters.ToArray();

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public NdArray ForwardCpu(NdArray x)
        {
            Real[][] upwards = new Real[4][];
            upwards[0] = this.upward0.Forward(x)[0].Data;
            upwards[1] = this.upward1.Forward(x)[0].Data;
            upwards[2] = this.upward2.Forward(x)[0].Data;
            upwards[3] = this.upward3.Forward(x)[0].Data;

            int outputDataSize = x.BatchCount * this.OutputCount;

            if (this.hParam == null)
            {
                //値がなければ初期化
                this.aParam = new List<Real[]>();
                this.iParam = new List<Real[]>();
                this.fParam = new List<Real[]>();
                this.oParam = new List<Real[]>();
                this.cParam = new List<Real[]>();
            }
            else
            {
                Real[] laterals0 = this.lateral0.Forward(hParam)[0].Data;
                Real[] laterals1 = this.lateral1.Forward(hParam)[0].Data;
                Real[] laterals2 = this.lateral2.Forward(hParam)[0].Data;
                Real[] laterals3 = this.lateral3.Forward(hParam)[0].Data;
                hParam.UseCount -= 4; //回数を補正 RFI

                for (int i = 0; i < outputDataSize; i++)
                {
                    upwards[0][i] += laterals0[i];
                    upwards[1][i] += laterals1[i];
                    upwards[2][i] += laterals2[i];
                    upwards[3][i] += laterals3[i];
                }
            }

            if (this.cParam.Count == 0)
            {
                this.cParam.Add(new Real[outputDataSize]);
            }

            Real[] la = new Real[outputDataSize];
            Real[] li = new Real[outputDataSize];
            Real[] lf = new Real[outputDataSize];
            Real[] lo = new Real[outputDataSize];
            Real[] cPrev = this.cParam[this.cParam.Count - 1];
            Real[] cResult = new Real[cPrev.Length];
            Real[] lhParam = new Real[outputDataSize];

            for (int b = 0; b < x.BatchCount; b++)
            {
                //再配置
                for (int j = 0; j < this.OutputCount; j++)
                {
                    int index = j * 4;
                    int batchIndex = b * OutputCount + j;

                    la[batchIndex] = Math.Tanh(upwards[index / this.OutputCount][index % this.OutputCount + b * OutputCount]);
                    li[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + b * OutputCount]);
                    lf[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + b * OutputCount]);
                    lo[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + b * OutputCount]);

                    cResult[batchIndex] = la[batchIndex] * li[batchIndex] + lf[batchIndex] * cPrev[batchIndex];

                    lhParam[batchIndex] = lo[batchIndex] * Math.Tanh(cResult[batchIndex]);
                }
            }

            //Backward用
            this.cParam.Add(cResult);
            this.aParam.Add(la);
            this.iParam.Add(li);
            this.fParam.Add(lf);
            this.oParam.Add(lo);

            this.hParam = new NdArray(lhParam, new[] { OutputCount }, x.BatchCount, this);
            return this.hParam;
        }

        public void BackwardCpu(NdArray y, NdArray x)
        {
            if (gcPrev == null)
            {
                //値がなければ初期化
                this.gxPrev0 = new NdArray(new[] { OutputCount }, y.BatchCount);
                this.gxPrev1 = new NdArray(new[] { OutputCount }, y.BatchCount);
                this.gxPrev2 = new NdArray(new[] { OutputCount }, y.BatchCount);
                this.gxPrev3 = new NdArray(new[] { OutputCount }, y.BatchCount);
                this.gcPrev = new Real[x.BatchCount * this.OutputCount];
            }
            else
            {
                this.lateral0.Backward(this.gxPrev0);
                this.lateral1.Backward(this.gxPrev1);
                this.lateral2.Backward(this.gxPrev2);
                this.lateral3.Backward(this.gxPrev3);
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

            for (int i = 0; i < y.BatchCount; i++)
            {
                Real[] gParam = new Real[this.InputCount * 4];

                for (int j = 0; j < this.InputCount; j++)
                {
                    int prevOutputIndex = j + i * this.OutputCount;
                    int prevInputIndex = j + i * this.InputCount;

                    double co = Math.Tanh(lcParam[prevOutputIndex]);

                    this.gcPrev[prevInputIndex] += y.Grad[prevOutputIndex] * loParam[prevOutputIndex] * GradTanh(co);
                    gParam[j + InputCount * 0] = this.gcPrev[prevInputIndex] * liParam[prevOutputIndex] * GradTanh(laParam[prevOutputIndex]);
                    gParam[j + InputCount * 1] = this.gcPrev[prevInputIndex] * laParam[prevOutputIndex] * GradSigmoid(liParam[prevOutputIndex]);
                    gParam[j + InputCount * 2] = this.gcPrev[prevInputIndex] * cPrev[prevOutputIndex] * GradSigmoid(lfParam[prevOutputIndex]);
                    gParam[j + InputCount * 3] = y.Grad[prevOutputIndex] * co * GradSigmoid(loParam[prevOutputIndex]);

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
                    this.gxPrev0.Grad[i * this.OutputCount + j] = resultParam[0 * this.OutputCount + j];
                    this.gxPrev1.Grad[i * this.OutputCount + j] = resultParam[1 * this.OutputCount + j];
                    this.gxPrev2.Grad[i * this.OutputCount + j] = resultParam[2 * this.OutputCount + j];
                    this.gxPrev3.Grad[i * this.OutputCount + j] = resultParam[3 * this.OutputCount + j];
                }
            }

            this.upward0.Backward(this.gxPrev0);
            this.upward1.Backward(this.gxPrev1);
            this.upward2.Backward(this.gxPrev2);
            this.upward3.Backward(this.gxPrev3);
        }

        public override void ResetState()
        {
            this.gcPrev = null;
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
