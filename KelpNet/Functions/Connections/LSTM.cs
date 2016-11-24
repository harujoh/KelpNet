using System;
using System.Collections.Generic;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

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

        private List<double[]>[] aParam;
        private List<double[]>[] iParam;
        private List<double[]>[] fParam;
        private List<double[]>[] oParam;
        private List<double[]>[] cParam;

        private NdArray[] hParam;

        private NdArray[] gxPrev0;
        private NdArray[] gxPrev1;
        private NdArray[] gxPrev2;
        private NdArray[] gxPrev3;
        private double[,] gcPrev;

        public LSTM(int inSize, int outSize, Array initialUpwardW = null, Array initialUpwardb = null, Array initialLateralW = null, string name = "LSTM") : base(name, inSize, outSize)
        {
            this.upward0 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward0");
            this.upward1 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward1");
            this.upward2 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward2");
            this.upward3 = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward3");
            this.Parameters.AddRange(this.upward0.Parameters);
            this.Parameters.AddRange(this.upward1.Parameters);
            this.Parameters.AddRange(this.upward2.Parameters);
            this.Parameters.AddRange(this.upward3.Parameters);

            //lateralはBiasは無し
            this.lateral0 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral0");
            this.lateral1 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral1");
            this.lateral2 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral2");
            this.lateral3 = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral3");
            this.Parameters.AddRange(this.lateral0.Parameters);
            this.Parameters.AddRange(this.lateral1.Parameters);
            this.Parameters.AddRange(this.lateral2.Parameters);
            this.Parameters.AddRange(this.lateral3.Parameters);
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] result = new NdArray[x.Length];

            NdArray[] upwards0 = this.upward0.Forward(x);
            NdArray[] upwards1 = this.upward1.Forward(x);
            NdArray[] upwards2 = this.upward2.Forward(x);
            NdArray[] upwards3 = this.upward3.Forward(x);

            if (this.hParam == null)
            {
                //値がなければ初期化
                this.InitBatch(x.Length);
            }
            else
            {
                //値があればupwardへ加算
                NdArray[] laterals0 = this.lateral0.Forward(this.hParam);
                NdArray[] laterals1 = this.lateral1.Forward(this.hParam);
                NdArray[] laterals2 = this.lateral2.Forward(this.hParam);
                NdArray[] laterals3 = this.lateral3.Forward(this.hParam);

                for (int j = 0; j < laterals0.Length; j++)
                {
                    for (int k = 0; k < laterals0[j].Length; k++)
                    {
                        upwards0[j].Data[k] += laterals0[j].Data[k];
                        upwards1[j].Data[k] += laterals1[j].Data[k];
                        upwards2[j].Data[k] += laterals2[j].Data[k];
                        upwards3[j].Data[k] += laterals3[j].Data[k];
                    }
                }
            }

#if DEBUG
            for (int i = 0; i < x.Length; i++)
#else
            Parallel.For(0, x.Length, i =>
#endif
            {
                if (this.cParam[i].Count == 0)
                {
                    this.cParam[i].Add(new double[this.OutputCount]);
                }

                //再配置
                double[,] r = this.ExtractGates(upwards0[i].Data, upwards1[i].Data, upwards2[i].Data, upwards3[i].Data);

                double[] la = new double[this.OutputCount];
                double[] li = new double[this.OutputCount];
                double[] lf = new double[this.OutputCount];
                double[] lo = new double[this.OutputCount];
                double[] cPrev = this.cParam[i][this.cParam[i].Count - 1];
                double[] cResult = new double[cPrev.Length];

                for (int j = 0; j < this.hParam[i].Length; j++)
                {
                    la[j] = Math.Tanh(r[0, j]);
                    li[j] = Sigmoid(r[1, j]);
                    lf[j] = Sigmoid(r[2, j]);
                    lo[j] = Sigmoid(r[3, j]);

                    cResult[j] = la[j] * li[j] + lf[j] * cPrev[j];
                    this.hParam[i].Data[j] = lo[j] * Math.Tanh(cResult[j]);
                }

                //Backward用
                this.cParam[i].Add(cResult);
                this.aParam[i].Add(la);
                this.iParam[i].Add(li);
                this.fParam[i].Add(lf);
                this.oParam[i].Add(lo);

                result[i] = this.hParam[i];
            }
#if !DEBUG
            );
#endif

            return result;
        }

        protected override NdArray[] BackwardSingle(NdArray[] gh)
        {
            NdArray[] result = new NdArray[gh.Length];

            if (this.gxPrev0 == null)
            {
                //値がなければ初期化
                this.gxPrev0 = new NdArray[gh.Length];
                this.gxPrev1 = new NdArray[gh.Length];
                this.gxPrev2 = new NdArray[gh.Length];
                this.gxPrev3 = new NdArray[gh.Length];
            }
            else
            {
                NdArray[] ghPre0 = this.lateral0.Backward(this.gxPrev0);
                NdArray[] ghPre1 = this.lateral1.Backward(this.gxPrev1);
                NdArray[] ghPre2 = this.lateral2.Backward(this.gxPrev2);
                NdArray[] ghPre3 = this.lateral3.Backward(this.gxPrev3);

                for (int j = 0; j < ghPre0.Length; j++)
                {
                    for (int k = 0; k < ghPre0[j].Length; k++)
                    {
                        gh[j].Data[k] += ghPre0[j].Data[k];
                        gh[j].Data[k] += ghPre1[j].Data[k];
                        gh[j].Data[k] += ghPre2[j].Data[k];
                        gh[j].Data[k] += ghPre3[j].Data[k];
                    }
                }
            }

#if DEBUG
            for (int i = 0; i < gh.Length; i++)
#else
            Parallel.For(0, gh.Length, i =>
#endif
            {
                double[] ga = new double[this.InputCount];
                double[] gi = new double[this.InputCount];
                double[] gf = new double[this.InputCount];
                double[] go = new double[this.InputCount];

                double[] lcParam = this.cParam[i][this.cParam[i].Count - 1];
                this.cParam[i].RemoveAt(this.cParam[i].Count - 1);

                double[] laParam = this.aParam[i][this.aParam[i].Count - 1];
                this.aParam[i].RemoveAt(this.aParam[i].Count - 1);

                double[] liParam = this.iParam[i][this.iParam[i].Count - 1];
                this.iParam[i].RemoveAt(this.iParam[i].Count - 1);

                double[] lfParam = this.fParam[i][this.fParam[i].Count - 1];
                this.fParam[i].RemoveAt(this.fParam[i].Count - 1);

                double[] loParam = this.oParam[i][this.oParam[i].Count - 1];
                this.oParam[i].RemoveAt(this.oParam[i].Count - 1);

                double[] cPrev = this.cParam[i][this.cParam[i].Count-1];

                for (int j = 0; j < this.InputCount; j++)
                {
                    double co = Math.Tanh(lcParam[j]);

                    this.gcPrev[i, j] = gh[i].Data[j] * loParam[j] * GradTanh(co) + this.gcPrev[i, j];
                    ga[j] = this.gcPrev[i, j] * liParam[j] * GradTanh(laParam[j]);
                    gi[j] = this.gcPrev[i, j] * laParam[j] * GradSigmoid(liParam[j]);
                    gf[j] = this.gcPrev[i, j] * cPrev[j] * GradSigmoid(lfParam[j]);
                    go[j] = gh[i].Data[j] * co * GradSigmoid(loParam[j]);

                    this.gcPrev[i, j] *= lfParam[j];
                }

                NdArray[] r = this.RestoreGates(ga, gi, gf, go);

                this.gxPrev0[i] = r[0];
                this.gxPrev1[i] = r[1];
                this.gxPrev2[i] = r[2];
                this.gxPrev3[i] = r[3];
            }
#if !DEBUG
            );
#endif

            NdArray[] gArray0 = this.upward0.Backward(this.gxPrev0);
            NdArray[] gArray1 = this.upward1.Backward(this.gxPrev1);
            NdArray[] gArray2 = this.upward2.Backward(this.gxPrev2);
            NdArray[] gArray3 = this.upward3.Backward(this.gxPrev3);

#if DEBUG
            for (int i = 0; i < gh.Length; i++)
#else
            Parallel.For(0, gh.Length, i =>
#endif
            {
                double[] gx = new double[this.InputCount];

                for (int j = 0; j < gx.Length; j++)
                {
                    gx[j] += gArray0[i].Data[j];
                    gx[j] += gArray1[i].Data[j];
                    gx[j] += gArray2[i].Data[j];
                    gx[j] += gArray3[i].Data[j];
                }

                result[i] = NdArray.FromArray(gx);
            }
#if !DEBUG
            );
#endif

            return result;
        }

        public override void ResetState()
        {
            this.hParam = null;
            this.gxPrev0 = null;
            this.gxPrev1 = null;
            this.gxPrev2 = null;
            this.gxPrev3 = null;
        }

        //バッチ実行時にバッティングするメンバをバッチ数分用意
        void InitBatch(int batchCount)
        {
            this.aParam = new List<double[]>[batchCount];
            this.iParam = new List<double[]>[batchCount];
            this.fParam = new List<double[]>[batchCount];
            this.oParam = new List<double[]>[batchCount];
            this.cParam = new List<double[]>[batchCount];
            this.hParam = new NdArray[batchCount];
            this.gcPrev = new double[batchCount, this.InputCount];

            for (int i = 0; i < batchCount; i++)
            {
                this.aParam[i] = new List<double[]>();
                this.iParam[i] = new List<double[]>();
                this.fParam[i] = new List<double[]>();
                this.oParam[i] = new List<double[]>();
                this.cParam[i] = new List<double[]>();
                this.hParam[i] = NdArray.Zeros(this.OutputCount);
            }
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

        //Forward用
        double[,] ExtractGates(params double[][] x)
        {
            double[,] r = new double[4, this.OutputCount];

            for (int i = 0; i < this.OutputCount; i++)
            {
                int index = i * 4;

                r[0, i] = x[index / this.OutputCount][index % this.OutputCount];
                r[1, i] = x[++index / this.OutputCount][index % this.OutputCount];
                r[2, i] = x[++index / this.OutputCount][index % this.OutputCount];
                r[3, i] = x[++index / this.OutputCount][index % this.OutputCount];
            }

            return r;
        }

        //Backward用
        NdArray[] RestoreGates(params double[][] x)
        {
            NdArray[] result =
            {
                NdArray.Zeros(this.OutputCount),
                NdArray.Zeros(this.OutputCount),
                NdArray.Zeros(this.OutputCount),
                NdArray.Zeros(this.OutputCount)
            };

            for (int i = 0; i < this.OutputCount * 4; i++)
            {
                //暗黙的に切り捨て
                result[i / this.OutputCount].Data[i % this.OutputCount] = x[i % 4][i / 4];
            }

            return result;
        }
    }
}
