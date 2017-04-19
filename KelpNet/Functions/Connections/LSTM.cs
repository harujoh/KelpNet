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

        BatchArray gxPrev0;
        BatchArray gxPrev1;
        BatchArray gxPrev2;
        BatchArray gxPrev3;
        Real[] gcPrev;

        public LSTM(int inSize, int outSize, Real[,] initialUpwardW = null, Real[] initialUpwardb = null, Real[,] initialLateralW = null, string name = "LSTM", bool isGpu = true) : base(name, isGpu, inSize, outSize)
        {
            this.Parameters = new FunctionParameter[12];

            this.upward0 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, isGpu: false, name: "upward0");
            this.upward1 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, isGpu: false, name: "upward1");
            this.upward2 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, isGpu: false, name: "upward2");
            this.upward3 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, isGpu: false, name: "upward3");
            this.Parameters[0] = this.upward0.Parameters[0];
            this.Parameters[1] = this.upward0.Parameters[1];
            this.Parameters[2] = this.upward1.Parameters[0];
            this.Parameters[3] = this.upward1.Parameters[1];
            this.Parameters[4] = this.upward2.Parameters[0];
            this.Parameters[5] = this.upward2.Parameters[1];
            this.Parameters[6] = this.upward3.Parameters[0];
            this.Parameters[7] = this.upward3.Parameters[1];

            //lateralはBiasは無し
            this.lateral0 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, isGpu: false, name: "lateral0");
            this.lateral1 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, isGpu: false, name: "lateral1");
            this.lateral2 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, isGpu: false, name: "lateral2");
            this.lateral3 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, isGpu: false, name: "lateral3");
            this.Parameters[8] = this.lateral0.Parameters[0];
            this.Parameters[9] = this.lateral1.Parameters[0];
            this.Parameters[10] = this.lateral2.Parameters[0];
            this.Parameters[11] = this.lateral3.Parameters[0];
        }

        public override void InitKernel()
        {
        }

        protected override BatchArray ForwardSingle(BatchArray x)
        {
            BatchArray[] upwards = new BatchArray[4];
            upwards[0] = this.upward0.Forward(x);
            upwards[1] = this.upward1.Forward(x);
            upwards[2] = this.upward2.Forward(x);
            upwards[3] = this.upward3.Forward(x);

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
                BatchArray prevInput = new BatchArray(this.hParam, new[] { OutputCount }, x.BatchCount);
                BatchArray laterals0 = this.lateral0.Forward(prevInput);
                BatchArray laterals1 = this.lateral1.Forward(prevInput);
                BatchArray laterals2 = this.lateral2.Forward(prevInput);
                BatchArray laterals3 = this.lateral3.Forward(prevInput);

                for (int i = 0; i < OutputCount * x.BatchCount; i++)
                {
                    upwards[0].Data[i] += laterals0.Data[i];
                    upwards[1].Data[i] += laterals1.Data[i];
                    upwards[2].Data[i] += laterals2.Data[i];
                    upwards[3].Data[i] += laterals3.Data[i];
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

                    la[batchIndex] = (Real)Math.Tanh(upwards[index / this.OutputCount].Data[index % this.OutputCount + i * OutputCount]);
                    li[batchIndex] = Sigmoid(upwards[++index / this.OutputCount].Data[index % this.OutputCount + i * OutputCount]);
                    lf[batchIndex] = Sigmoid(upwards[++index / this.OutputCount].Data[index % this.OutputCount + i * OutputCount]);
                    lo[batchIndex] = Sigmoid(upwards[++index / this.OutputCount].Data[index % this.OutputCount + i * OutputCount]);

                    cResult[batchIndex] = la[batchIndex] * li[batchIndex] + lf[batchIndex] * cPrev[batchIndex];
                    this.hParam[batchIndex] = lo[batchIndex] * (Real)Math.Tanh(cResult[batchIndex]);
                }
            }

            //Backward用
            this.cParam.Add(cResult);
            this.aParam.Add(la);
            this.iParam.Add(li);
            this.fParam.Add(lf);
            this.oParam.Add(lo);

            return new BatchArray(this.hParam, new[] { OutputCount }, x.BatchCount);
        }

        protected override BatchArray BackwardSingle(BatchArray gh)
        {
            Real[] lgh = gh.Data.ToArray();

            if (this.gxPrev0 == null)
            {
                //値がなければ初期化
                this.gxPrev0 = new BatchArray(new[] { OutputCount }, gh.BatchCount);
                this.gxPrev1 = new BatchArray(new[] { OutputCount }, gh.BatchCount);
                this.gxPrev2 = new BatchArray(new[] { OutputCount }, gh.BatchCount);
                this.gxPrev3 = new BatchArray(new[] { OutputCount }, gh.BatchCount);
            }
            else
            {
                BatchArray ghPre0 = this.lateral0.Backward(this.gxPrev0);
                BatchArray ghPre1 = this.lateral1.Backward(this.gxPrev1);
                BatchArray ghPre2 = this.lateral2.Backward(this.gxPrev2);
                BatchArray ghPre3 = this.lateral3.Backward(this.gxPrev3);

                for (int j = 0; j < gh.BatchCount * OutputCount; j++)
                {
                    lgh[j] += ghPre0.Data[j];
                    lgh[j] += ghPre1.Data[j];
                    lgh[j] += ghPre2.Data[j];
                    lgh[j] += ghPre3.Data[j];
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

                    Real co = (Real)Math.Tanh(lcParam[prevOutputIndex]);

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

            BatchArray gArray0 = this.upward0.Backward(this.gxPrev0);
            BatchArray gArray1 = this.upward1.Backward(this.gxPrev1);
            BatchArray gArray2 = this.upward2.Backward(this.gxPrev2);
            BatchArray gArray3 = this.upward3.Backward(this.gxPrev3);

            Real[] gx = new Real[gh.BatchCount * this.InputCount];

            for (int i = 0; i < gx.Length; i++)
            {
                gx[i] = gArray0.Data[i] + gArray1.Data[i] + gArray2.Data[i] + gArray3.Data[i];
            }

            return BatchArray.Convert(gx, new[] { InputCount }, gh.BatchCount);
        }

        public override void ResetState()
        {
            this.hParam = null;
            this.gxPrev0 = null;
            this.gxPrev1 = null;
            this.gxPrev2 = null;
            this.gxPrev3 = null;
        }

        static Real Sigmoid(Real x)
        {
            return 1f / (1f + (Real)Math.Exp(-x));
        }

        static Real GradSigmoid(Real x)
        {
            return x * (1f - x);
        }

        static Real GradTanh(Real x)
        {
            return 1f - x * x;
        }
    }
}
