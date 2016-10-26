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
        private List<double[]>[] aParam;
        private List<double[]>[] iParam;
        private List<double[]>[] fParam;
        private List<double[]>[] oParam;
        private List<double[]>[] cParam; // = new List<double[]>[1];

        private NdArray[] hParam;

        public Linear[] upward = new Linear[4];
        public Linear[] lateral = new Linear[4];

        private NdArray[][] gxPrev;
        private double[][] gcPrev;

        public LSTM(int inSize, int outSize, Array initialUpwardW = null, Array initialUpwardb = null, Array initialLateralW = null, string name = "LSTM") : base(name)
        {
            for (int i = 0; i < 4; i++)
            {
                this.upward[i] = new Linear(inSize, outSize, noBias: false, initialW: initialUpwardW, initialb: initialUpwardb, name: "upward" + i);
                Parameters.AddRange(this.upward[i].Parameters);

                //lateralはBiasは無し
                this.lateral[i] = new Linear(outSize, outSize, noBias: true, initialW: initialLateralW, name: "lateral" + i);
                Parameters.AddRange(this.lateral[i].Parameters);
            }

            InputCount = inSize;
            OutputCount = outSize;
        }

        protected override NdArray[] ForwardSingle(NdArray[] x)
        {
            NdArray[] result = new NdArray[x.Length];
            NdArray[][] upwards = new NdArray[4][];
            NdArray[][] laterals = new NdArray[4][];

            for (int i = 0; i < upwards.Length; i++)
            {
                upwards[i] = this.upward[i].Forward(x);
            }

            if (this.hParam != null)
            {
                for (int i = 0; i < laterals.Length; i++)
                {
                    laterals[i] = this.lateral[i].Forward(this.hParam);
                }
            }
            else
            {
                this.InitBatch(x.Length);

                this.hParam = new NdArray[x.Length];
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
                Array.Copy(upwards[0][i].Data, 0, upwardResult, 0 * OutputCount, OutputCount);
                Array.Copy(upwards[1][i].Data, 0, upwardResult, 1 * OutputCount, OutputCount);
                Array.Copy(upwards[2][i].Data, 0, upwardResult, 2 * OutputCount, OutputCount);
                Array.Copy(upwards[3][i].Data, 0, upwardResult, 3 * OutputCount, OutputCount);

                double[][] r;
                if (this.hParam[i] != null)
                {
                    double[] lateralResult = new double[OutputCount * 4];
                    Array.Copy(laterals[0][i].Data, 0, lateralResult, 0 * OutputCount, OutputCount);
                    Array.Copy(laterals[1][i].Data, 0, lateralResult, 1 * OutputCount, OutputCount);
                    Array.Copy(laterals[2][i].Data, 0, lateralResult, 2 * OutputCount, OutputCount);
                    Array.Copy(laterals[3][i].Data, 0, lateralResult, 3 * OutputCount, OutputCount);

                    //加算しつつ再配置
                    r = this.ExtractGates(upwardResult, lateralResult);
                }
                else
                {
                    this.hParam[i] = NdArray.Zeros(OutputCount);

                    r = this.ExtractGates(upwardResult);
                }

                var la = new double[OutputCount];
                var li = new double[OutputCount];
                var lf = new double[OutputCount];
                var lo = new double[OutputCount];
                var cPrev = this.cParam[i].Last();
                var cResult = new double[cPrev.Length];

                for (int j = 0; j < this.hParam[i].Length; j++)
                {
                    la[j] = Math.Tanh(r[0][j]);
                    li[j] = Sigmoid(r[1][j]);
                    lf[j] = Sigmoid(r[2][j]);
                    lo[j] = Sigmoid(r[3][j]);

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
                NdArray[][] ghPre = new NdArray[4][];

                for (int i = 0; i < ghPre.Length; i++)
                {
                    ghPre[i] = this.lateral[i].Backward(this.gxPrev[i]);

                    for (int j = 0; j < ghPre[i].Length; j++)
                    {
                        for (int k = 0; k < ghPre[i][j].Length; k++)
                        {
                            gh[j].Data[k] += ghPre[i][j].Data[k];
                        }
                    }
                }
            }
            else
            {
                this.gxPrev = new []
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

            NdArray[][] gArray = new NdArray[4][];
            for (int i = 0; i < gArray.Length; i++)
            {
                gArray[i] = this.upward[i].Backward(this.gxPrev[i]);
            }

#if DEBUG
            for (int i = 0; i < gh.Length; i++)
#else
            Parallel.For(0, gh.Length, i =>
#endif
            {
                double[] gx = new double[InputCount];

                for (int j = 0; j < gx.Length; j++)
                {
                    for (int k = 0; k < gArray.Length; k++)
                    {
                        gx[j] += gArray[k][i].Data[j];
                    }
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
        double[][] ExtractGates(params double[][] x)
        {
            int col = x[0].Length / 4;

            double[][] r =
            {
                new double[col],
                new double[col],
                new double[col],
                new double[col]
            };

            for (int j = 0; j < x.Length; j++)
            {
                for (int i = 0; i < col; i++)
                {
                    r[0][i] += x[j][i * 4];
                    r[1][i] += x[j][i * 4 + 1];
                    r[2][i] += x[j][i * 4 + 2];
                    r[3][i] += x[j][i * 4 + 3];
                }
            }

            return r;
        }

        //Backward用
        NdArray[] RestoreGates(params double[][] x)
        {
            int col = x[0].Length;

            NdArray[] result =
            {
                NdArray.Zeros(col),
                NdArray.Zeros(col),
                NdArray.Zeros(col),
                NdArray.Zeros(col)
            };

            for (int i = 0; i < col * 4; i++)
            {
                //暗黙的に切り捨て
                result[i / col].Data[i % col] = x[i % 4][i / 4];
            }

            return result;
        }
    }
}
