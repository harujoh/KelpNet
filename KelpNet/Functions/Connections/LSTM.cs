using System;
using System.Collections.Generic;

namespace KelpNet
{
    [Serializable]
    public class LSTM<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "LSTM";

        public Linear<T> upward0;
        public Linear<T> upward1;
        public Linear<T> upward2;
        public Linear<T> upward3;

        public Linear<T> lateral0;
        public Linear<T> lateral1;
        public Linear<T> lateral2;
        public Linear<T> lateral3;

        private List<Real<T>[]> aParam;
        private List<Real<T>[]> iParam;
        private List<Real<T>[]> fParam;
        private List<Real<T>[]> oParam;
        private List<Real<T>[]> cParam;

        private NdArray<T> hParam;
        private bool Initialized;

        NdArray<T> gxPrev0;
        NdArray<T> gxPrev1;
        NdArray<T> gxPrev2;
        NdArray<T> gxPrev3;
        Real<T>[] gcPrev;

        public readonly int InputCount;
        public readonly int OutputCount;

        public LSTM(int inSize, int outSize, Real<T>[,] initialUpwardW = null, Real<T>[] initialUpwardb = null, Real<T>[,] initialLateralW = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null, bool gpuEnable = false) : base(name, inputNames, outputNames)
        {
            this.InputCount = inSize;
            this.OutputCount = outSize;

            List<NdArray<T>> functionParameters = new List<NdArray<T>>();

            this.upward0 = new Linear<T>(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward0");
            this.upward1 = new Linear<T>(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward1");
            this.upward2 = new Linear<T>(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward2");
            this.upward3 = new Linear<T>(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward3");

            functionParameters.AddRange(this.upward0.Parameters);
            functionParameters.AddRange(this.upward1.Parameters);
            functionParameters.AddRange(this.upward2.Parameters);
            functionParameters.AddRange(this.upward3.Parameters);

            //lateralはBiasは無し
            this.lateral0 = new Linear<T>(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral0");
            this.lateral1 = new Linear<T>(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral1");
            this.lateral2 = new Linear<T>(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral2");
            this.lateral3 = new Linear<T>(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral3");

            functionParameters.AddRange(this.lateral0.Parameters);
            functionParameters.AddRange(this.lateral1.Parameters);
            functionParameters.AddRange(this.lateral2.Parameters);
            functionParameters.AddRange(this.lateral3.Parameters);

            this.Parameters = functionParameters.ToArray();

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public NdArray<T> ForwardCpu(NdArray<T> x)
        {
            Real<T>[][] upwards = new Real<T>[4][];
            upwards[0] = this.upward0.Forward(x)[0].Data;
            upwards[1] = this.upward1.Forward(x)[0].Data;
            upwards[2] = this.upward2.Forward(x)[0].Data;
            upwards[3] = this.upward3.Forward(x)[0].Data;

            int outputDataSize = x.BatchCount * this.OutputCount;

            if(!Initialized)
            {
                //値がなければ初期化
                this.aParam = new List<Real<T>[]>();
                this.iParam = new List<Real<T>[]>();
                this.fParam = new List<Real<T>[]>();
                this.oParam = new List<Real<T>[]>();
                this.cParam = new List<Real<T>[]>();
            }
            else
            {
                Real<T>[] laterals0 = this.lateral0.Forward(hParam)[0].Data;
                Real<T>[] laterals1 = this.lateral1.Forward(hParam)[0].Data;
                Real<T>[] laterals2 = this.lateral2.Forward(hParam)[0].Data;
                Real<T>[] laterals3 = this.lateral3.Forward(hParam)[0].Data;
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
                this.cParam.Add(new Real<T>[outputDataSize]);
            }

            Real<T>[] la = new Real<T>[outputDataSize];
            Real<T>[] li = new Real<T>[outputDataSize];
            Real<T>[] lf = new Real<T>[outputDataSize];
            Real<T>[] lo = new Real<T>[outputDataSize];
            Real<T>[] cPrev = this.cParam[this.cParam.Count - 1];
            Real<T>[] cResult = new Real<T>[cPrev.Length];
            Real<T>[] lhParam = new Real<T>[outputDataSize];

            for (int b = 0; b < x.BatchCount; b++)
            {
                //再配置
                for (int j = 0; j < this.OutputCount; j++)
                {
                    int index = j * 4;
                    int batchIndex = b * OutputCount + j;

                    la[batchIndex] = (Real<T>)Math.Tanh(upwards[index / this.OutputCount][index % this.OutputCount + b * OutputCount]);
                    li[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + b * OutputCount]);
                    lf[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + b * OutputCount]);
                    lo[batchIndex] = Sigmoid(upwards[++index / this.OutputCount][index % this.OutputCount + b * OutputCount]);

                    cResult[batchIndex] = la[batchIndex] * li[batchIndex] + lf[batchIndex] * cPrev[batchIndex];

                    lhParam[batchIndex] = lo[batchIndex] * (Real<T>)Math.Tanh(cResult[batchIndex]);
                }
            }

            //Backward用
            this.cParam.Add(cResult);
            this.aParam.Add(la);
            this.iParam.Add(li);
            this.fParam.Add(lf);
            this.oParam.Add(lo);

            Initialized = true;
            this.hParam = new NdArray<T>(lhParam, new[] { OutputCount }, x.BatchCount, this);
            return this.hParam;
        }

        public void BackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            if (gcPrev == null)
            {
                //値がなければ初期化
                this.gxPrev0 = new NdArray<T>(new[] { OutputCount }, y.BatchCount);
                this.gxPrev1 = new NdArray<T>(new[] { OutputCount }, y.BatchCount);
                this.gxPrev2 = new NdArray<T>(new[] { OutputCount }, y.BatchCount);
                this.gxPrev3 = new NdArray<T>(new[] { OutputCount }, y.BatchCount);
                this.gcPrev = new Real<T>[x.BatchCount * this.OutputCount];
            }
            else
            {
                this.lateral0.Backward(this.gxPrev0);
                this.lateral1.Backward(this.gxPrev1);
                this.lateral2.Backward(this.gxPrev2);
                this.lateral3.Backward(this.gxPrev3);
            }

            Real<T>[] lcParam = this.cParam[this.cParam.Count - 1];
            this.cParam.RemoveAt(this.cParam.Count - 1);

            Real<T>[] laParam = this.aParam[this.aParam.Count - 1];
            this.aParam.RemoveAt(this.aParam.Count - 1);

            Real<T>[] liParam = this.iParam[this.iParam.Count - 1];
            this.iParam.RemoveAt(this.iParam.Count - 1);

            Real<T>[] lfParam = this.fParam[this.fParam.Count - 1];
            this.fParam.RemoveAt(this.fParam.Count - 1);

            Real<T>[] loParam = this.oParam[this.oParam.Count - 1];
            this.oParam.RemoveAt(this.oParam.Count - 1);

            Real<T>[] cPrev = this.cParam[this.cParam.Count - 1];

            for (int i = 0; i < y.BatchCount; i++)
            {
                Real<T>[] gParam = new Real<T>[this.InputCount * 4];

                for (int j = 0; j < this.InputCount; j++)
                {
                    int prevOutputIndex = j + i * this.OutputCount;
                    int prevInputIndex = j + i * this.InputCount;

                    Real<T> co = (Real<T>)Math.Tanh(lcParam[prevOutputIndex]);

                    this.gcPrev[prevInputIndex] += y.Grad[prevOutputIndex] * loParam[prevOutputIndex] * GradTanh(co);
                    gParam[j + InputCount * 0] = this.gcPrev[prevInputIndex] * liParam[prevOutputIndex] * GradTanh(laParam[prevOutputIndex]);
                    gParam[j + InputCount * 1] = this.gcPrev[prevInputIndex] * laParam[prevOutputIndex] * GradSigmoid(liParam[prevOutputIndex]);
                    gParam[j + InputCount * 2] = this.gcPrev[prevInputIndex] * cPrev[prevOutputIndex] * GradSigmoid(lfParam[prevOutputIndex]);
                    gParam[j + InputCount * 3] = y.Grad[prevOutputIndex] * co * GradSigmoid(loParam[prevOutputIndex]);

                    this.gcPrev[prevInputIndex] *= lfParam[prevOutputIndex];
                }

                Real<T>[] resultParam = new Real<T>[this.OutputCount * 4];

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
            base.ResetState();
            this.gcPrev = new Real<T>[]{};
            this.hParam = new NdArray<T>();
            this.Initialized = false;
        }

        static Real<T> Sigmoid(Real<T> x)
        {
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        static Real<T> GradSigmoid(Real<T> x)
        {
            return (x * (1.0 - x));
        }

        static Real<T> GradTanh(Real<T> x)
        {
            return (1.0 - x * x);
        }
    }
}
