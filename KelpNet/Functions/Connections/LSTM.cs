using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class LSTM : Function
    {
        public Linear[] upward = new Linear[4];
        public Linear[] lateral = new Linear[4];

        private List<double[]>[] aParam;
        private List<double[]>[] iParam;
        private List<double[]>[] fParam;
        private List<double[]>[] oParam;
        private List<double[]>[] cParam;

        private NdArray[] hParam;

        private NdArray[][] gxPrev;
        private double[][] gcPrev;

        public LSTM(int inSize, int outSize, Array initialUpwardW = null, Array initialUpwardb = null, Array initialLateralW = null, string name = "LSTM") : base(name, inSize, outSize)
        {
            for (int i = 0; i < 4; i++)
            {
                this.upward[i] = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward" + i);
                Parameters.AddRange(this.upward[i].Parameters);

                //lateralはBiasは無し
                this.lateral[i] = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral" + i);
                Parameters.AddRange(this.lateral[i].Parameters);
            }
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] result = new NdArray[x.Length];

            var upwards0 = this.upward[0].Forward(x);
            var upwards1 = this.upward[1].Forward(x);
            var upwards2 = this.upward[2].Forward(x);
            var upwards3 = this.upward[3].Forward(x);

            if (this.hParam == null)
            {
                //値がなければ初期化
                this.InitBatch(x.Length);

                this.hParam = new NdArray[x.Length];
                for (int i = 0; i < this.hParam.Length; i++)
                {
                    this.hParam[i] = NdArray.Zeros(OutputCount);
                }
            }
            else
            {
                //値があればupwardへ加算
                var laterals0 = this.lateral[0].Forward(this.hParam);
                var laterals1 = this.lateral[1].Forward(this.hParam);
                var laterals2 = this.lateral[2].Forward(this.hParam);
                var laterals3 = this.lateral[3].Forward(this.hParam);

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
                    this.cParam[i].Add(new double[OutputCount]);
                }

                double[] upwardResult = new double[OutputCount * 4];
                Buffer.BlockCopy(upwards0[i].Data, 0, upwardResult, sizeof(double) * 0 * OutputCount, sizeof(double) * OutputCount);
                Buffer.BlockCopy(upwards1[i].Data, 0, upwardResult, sizeof(double) * 1 * OutputCount, sizeof(double) * OutputCount);
                Buffer.BlockCopy(upwards2[i].Data, 0, upwardResult, sizeof(double) * 2 * OutputCount, sizeof(double) * OutputCount);
                Buffer.BlockCopy(upwards3[i].Data, 0, upwardResult, sizeof(double) * 3 * OutputCount, sizeof(double) * OutputCount);

                //再配置
                double[,] r = this.ExtractGates(upwardResult);

                var la = new double[OutputCount];
                var li = new double[OutputCount];
                var lf = new double[OutputCount];
                var lo = new double[OutputCount];
                var cPrev = this.cParam[i].Last();
                var cResult = new double[cPrev.Length];

                for (int j = 0; j < this.hParam[i].Length; j++)
                {
                    la[j] = Math.Tanh(r[0,j]);
                    li[j] = Sigmoid(r[1,j]);
                    lf[j] = Sigmoid(r[2,j]);
                    lo[j] = Sigmoid(r[3,j]);

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

            if (this.gxPrev != null)
            {
                var ghPre0 = this.lateral[0].Backward(this.gxPrev[0]);
                var ghPre1 = this.lateral[1].Backward(this.gxPrev[1]);
                var ghPre2 = this.lateral[2].Backward(this.gxPrev[2]);
                var ghPre3 = this.lateral[3].Backward(this.gxPrev[3]);

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
            else
            {
                this.gxPrev = new[]
                {
                    new NdArray[gh.Length],
                    new NdArray[gh.Length],
                    new NdArray[gh.Length],
                    new NdArray[gh.Length],
                };
            }

#if DEBUG
            for (int i = 0; i < gh.Length; i++)
#else
            Parallel.For(0, gh.Length, i =>
#endif
            {
                if (this.gcPrev[i] == null)
                {
                    this.gcPrev[i] = new double[InputCount];
                }

                var ga = new double[InputCount];
                var gi = new double[InputCount];
                var gf = new double[InputCount];
                var go = new double[InputCount];

                var lcParam = this.cParam[i].Last();
                var laParam = this.aParam[i].Last();
                var liParam = this.iParam[i].Last();
                var lfParam = this.fParam[i].Last();
                var loParam = this.oParam[i].Last();

                this.cParam[i].RemoveAt(this.cParam[i].Count - 1);
                this.aParam[i].RemoveAt(this.aParam[i].Count - 1);
                this.iParam[i].RemoveAt(this.iParam[i].Count - 1);
                this.fParam[i].RemoveAt(this.fParam[i].Count - 1);
                this.oParam[i].RemoveAt(this.oParam[i].Count - 1);

                var cPrev = this.cParam[i].Last();

                for (int j = 0; j < this.gcPrev[i].Length; j++)
                {
                    var co = Math.Tanh(lcParam[j]);

                    this.gcPrev[i][j] = gh[i].Data[j] * loParam[j] * GradTanh(co) + this.gcPrev[i][j];
                    ga[j] = this.gcPrev[i][j] * liParam[j] * GradTanh(laParam[j]);
                    gi[j] = this.gcPrev[i][j] * laParam[j] * GradSigmoid(liParam[j]);
                    gf[j] = this.gcPrev[i][j] * cPrev[j] * GradSigmoid(lfParam[j]);
                    go[j] = gh[i].Data[j] * co * GradSigmoid(loParam[j]);

                    this.gcPrev[i][j] *= lfParam[j];
                }

                var r = this.RestoreGates(ga, gi, gf, go);
                this.gxPrev[0][i] = r[0];
                this.gxPrev[1][i] = r[1];
                this.gxPrev[2][i] = r[2];
                this.gxPrev[3][i] = r[3];
            }
#if !DEBUG
            );
#endif

            var gArray0 = this.upward[0].Backward(this.gxPrev[0]);
            var gArray1 = this.upward[1].Backward(this.gxPrev[1]);
            var gArray2 = this.upward[2].Backward(this.gxPrev[2]);
            var gArray3 = this.upward[3].Backward(this.gxPrev[3]);

#if DEBUG
            for (int i = 0; i < gh.Length; i++)
#else
            Parallel.For(0, gh.Length, i =>
#endif
            {
                double[] gx = new double[InputCount];

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
            this.cParam = new List<double[]>[this.cParam.Length];

            for (int i = 0; i < this.cParam.Length; i++)
            {
                this.cParam[i] = new List<double[]>();
            }

            this.hParam = null;

            this.gcPrev = new double[this.gcPrev.Length][];
            this.gxPrev = null;
        }

        //バッチ実行時にバッティングするメンバをバッチ数分用意
        void InitBatch(int batchCount)
        {
            this.aParam = new List<double[]>[batchCount];
            this.iParam = new List<double[]>[batchCount];
            this.fParam = new List<double[]>[batchCount];
            this.oParam = new List<double[]>[batchCount];
            this.cParam = new List<double[]>[batchCount];

            for (int i = 0; i < batchCount; i++)
            {
                this.aParam[i] = new List<double[]>();
                this.iParam[i] = new List<double[]>();
                this.fParam[i] = new List<double[]>();
                this.oParam[i] = new List<double[]>();
                this.cParam[i] = new List<double[]>();
            }

            this.gcPrev = new double[batchCount][];
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
        double[,] ExtractGates(double[] x)
        {
            double[,] r = new double[4,OutputCount];

            for (int i = 0; i < OutputCount; i++)
            {
                r[0,i] = x[i * 4 + 0];
                r[1,i] = x[i * 4 + 1];
                r[2,i] = x[i * 4 + 2];
                r[3,i] = x[i * 4 + 3];
            }

            return r;
        }

        //Backward用
        NdArray[] RestoreGates(params double[][] x)
        {
            NdArray[] result =
            {
                NdArray.Zeros(OutputCount),
                NdArray.Zeros(OutputCount),
                NdArray.Zeros(OutputCount),
                NdArray.Zeros(OutputCount)
            };

            for (int i = 0; i < OutputCount * 4; i++)
            {
                //暗黙的に切り捨て
                result[i / OutputCount].Data[i % OutputCount] = x[i % 4][i / 4];
            }

            return result;
        }
    }
}
