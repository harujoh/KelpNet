using System;
using System.Collections.Generic;
using KelpNet.Interface;

namespace KelpNet.Functions.Connections
{
    public class LSTM : Function, IPredictableFunction
    {
        public Stack<NdArray>[] aParam;
        public Stack<NdArray>[] iParam;
        public Stack<NdArray>[] fParam;
        public Stack<NdArray>[] oParam;
        public Stack<NdArray>[] cParam;

        private NdArray[] hParam = new NdArray[1];

        public Stack<NdArray>[] cPrev;
        public NdArray[] cSave = new NdArray[1];

        public Linear[] upward = new Linear[4];
        public Linear[] lateral = new Linear[4];

        private NdArray[][] gxPrev = new NdArray[1][];
        private NdArray[] gcPrev = new NdArray[1];

        public LSTM(int inSize, int outSize, string name = "LSTM") : base(name)
        {
#if DEBUG
            for (int i = 0; i < 4; i++)
            {
                this.upward[i] = new Linear(inSize, outSize, name: "upward" + i);

                //lateralはBiasは無し
                this.lateral[i] = new Linear(outSize, outSize, noBias: true, name: "lateral" + i);
            }

            this.upward[0].W = NdArray.FromArray(new[,]{
            {-0.04395561,  0.00953909, -0.08359507,  0.08964871, -0.00650719},
                {-0.0575492,   0.07147719,  0.08890767, -0.0564871,  -0.09726748},
                {-0.05821012,  0.0130387,   0.06219254,  0.03400032, -0.01163356},
                {-0.07131696, -0.03838185, -0.09704818,  0.02615761,  0.02697979},
                { 0.08959083, -0.07590719, -0.05582947,  0.02060409,  0.04031029}
            });
            this.upward[0].b = NdArray.FromArray(new[] { 0.03594227, -0.00667736, 0.00918686, -0.05687213, -0.09300639 });

            this.upward[1].W = NdArray.FromArray(new[,]{
                { 0.00493388,  0.01904213,  0.06349733,  0.08207227,  0.09927054},
                {-0.05937263, -0.0824149,  -0.06575577,  0.04905749,  0.06992707},
                {-0.00150587,  0.04190406,  0.02523944, -0.07424963,  0.05040478},
                {-0.09756977,  0.06557105, -0.0616635,  -0.04204918,  0.02662499},
                { 0.04577347, -0.01056622,  0.04013773, -0.04184886, -0.0828003 }
            });
            this.upward[1].b = NdArray.FromArray(new[] { 0.05552417, -0.01824856, -0.01015915, -0.04751314, -0.03874684 });

            this.upward[2].W = NdArray.FromArray(new[,]{
                {-0.06588724,  0.05582662, -0.07949429, -0.0348076,   0.0511508 },
                {-0.01625326, -0.02578919, -0.03175092,  0.03544928, -0.0100778 },
                {-0.07827286,  0.04940033,  0.08252732, -0.08152542, -0.0321954 },
                { 0.02304256, -0.0946193,   0.03883769,  0.07799184, -0.0723974 },
                {-0.06058234, -0.09420774,  0.06511135,  0.03198707, -0.03319233}
            });
            this.upward[2].b = NdArray.FromArray(new[] { -0.05932232, 0.08657682, 0.0953408, 0.0827532, 0.01947624 });

            this.upward[3].W = NdArray.FromArray(new[,]{
                {-0.0430833,   0.07852951,  0.08043411,  0.09334686, -0.04600554},
                { 0.04217525, -0.03100407, -0.03556751,  0.07804406, -0.09834558},
                { 0.0883133,  -0.02476237, -0.08914164,  0.0588782,   0.04889128},
                {-0.01575579, -0.00557019,  0.05506143,  0.03283211,  0.07986852},
                {-0.0146129,   0.02733113,  0.05563244,  0.04313883, -0.00209575}
            });
            this.upward[3].b = NdArray.FromArray(new[] { -0.06486291, 0.05479813, 0.00306977, 0.07728702, 0.04547007 });


            this.lateral[0].W = NdArray.FromArray(new[,]{
                { 0.00970838, -0.09677468, -0.05098628,  0.01430712, -0.06938247},
                {-0.06167605, -0.05409741, -0.02639686,  0.08743017, -0.06690538},
                {0.05643582, -0.03038138, -0.03113236, -0.03279318,  0.0631598},
                {-0.05885514, -0.07111458,  0.04224461, -0.09915958,  0.02011202},
                {0.07988876,  0.0377701,   0.03605221,  0.07979446, -0.05496476}
            });

            this.lateral[1].W = NdArray.FromArray(new[,]{
                {-0.01900506,  0.01345564,  0.07828248,  0.04267332, -0.00484286},
                {0.09092383,  0.02374125, -0.02484604, -0.01459825, -0.08989462},
                {0.04657019, -0.00915463,  0.00073637,  0.00318613, -0.08723193},
                {0.0380653,  -0.04503078,  0.0613486,  -0.00372761,  0.04565996},
                {-0.08838285, -0.04949837,  0.08403265, -0.09652784, -0.08095802}
            });

            this.lateral[2].W = NdArray.FromArray(new[,]{
                {0.05430992, -0.063843,    0.02891743,  0.06503975, -0.07874666},
                {-0.09891963, -0.07618238, -0.03567919, -0.06237401, -0.04461374},
                {-0.03283143,  0.09137988, -0.08329749, -0.05893804,  0.03652418},
                {-0.0428125,   0.02421166,  0.07261568,  0.06710947,  0.08136631},
                {-0.08614939, -0.05302165, -0.06242198, -0.00018274, -0.09131889}
            });

            this.lateral[3].W = NdArray.FromArray(new[,]{
                {-0.06869422, -0.09478083,  0.00827363,  0.08640816, -0.05355168},
                {-0.03474252,  0.06299275,  0.05079195, -0.00116762,  0.05746876},
                {-0.01644047, -0.09919241,  0.03294539,  0.0124337,   0.00149174},
                {-0.0728629,   0.02334053, -0.06079264,  0.0744803,  -0.02010363},
                {-0.08747777,  0.0438894,   0.06413155,  0.08054566,  0.01802104}
            });

            for (int i = 0; i < 4; i++)
            {
                Parameters.Add(new OptimizeParameter(this.upward[i].W, this.upward[i].gW, this.Name + " " + this.upward[i].Name + " " + " W"));
                Parameters.Add(new OptimizeParameter(this.upward[i].b, this.upward[i].gb, this.Name + " " + this.upward[i].Name + " " + " b"));

                //lateralはBiasは無し
                Parameters.Add(new OptimizeParameter(this.lateral[i].W, this.lateral[i].gW, this.Name + " " + this.lateral[i].Name + " " + " W"));
            }
#else
            for (int i = 0; i < 4; i++)
            {
                this.upward[i] = new Linear(inSize, outSize, name: "upward" + i);
                Parameters.Add(new OptimizeParameter(this.upward[i].W, this.upward[i].gW, this.Name + " " + this.upward[i].Name + " " +  " W"));
                Parameters.Add(new OptimizeParameter(this.upward[i].b, this.upward[i].gb, this.Name + " " + this.upward[i].Name + " " +  " b"));

                //lateralはBiasは無し
                this.lateral[i] = new Linear(outSize, outSize, noBias: true, name: "lateral" + i);
                Parameters.Add(new OptimizeParameter(this.lateral[i].W, this.lateral[i].gW, this.Name + " " + this.lateral[i].Name + " " +  " W"));
            }
#endif
        }

        protected override NdArray ForwardSingle(NdArray x, int batchID = 0)
        {
            if (this.cSave[batchID] == null)
            {
                this.cSave[batchID] = NdArray.ZerosLike(x);
            }

            this.cPrev[batchID].Push(new NdArray(this.cSave[batchID]));

            List<double> tmp = new List<double>();
            tmp.AddRange(this.upward[0].Forward(x, batchID).Data);
            tmp.AddRange(this.upward[1].Forward(x, batchID).Data);
            tmp.AddRange(this.upward[2].Forward(x, batchID).Data);
            tmp.AddRange(this.upward[3].Forward(x, batchID).Data);

            NdArray[] r;
            if (this.hParam[batchID] != null)
            {
                List<double> tmp2 = new List<double>();
                tmp2.AddRange(this.lateral[0].Forward(this.hParam[batchID], batchID).Data);
                tmp2.AddRange(this.lateral[1].Forward(this.hParam[batchID], batchID).Data);
                tmp2.AddRange(this.lateral[2].Forward(this.hParam[batchID], batchID).Data);
                tmp2.AddRange(this.lateral[3].Forward(this.hParam[batchID], batchID).Data);
                r = this.ExtractGates(tmp, tmp2);
            }
            else
            {
                r = this.ExtractGates(tmp);
                this.hParam[batchID] = NdArray.ZerosLike(x);
            }

            var la = r[0];
            var li = r[1];
            var lf = r[2];
            var lo = r[3];

            var laParam = NdArray.EmptyLike(la);
            var liParam = NdArray.EmptyLike(li);
            var lfParam = NdArray.EmptyLike(lf);
            var loParam = NdArray.EmptyLike(lo);

            for (int i = 0; i < this.hParam[batchID].Length; i++)
            {
                laParam.Data[i] = Math.Tanh(la.Data[i]);
                liParam.Data[i] = Sigmoid(li.Data[i]);
                lfParam.Data[i] = Sigmoid(lf.Data[i]);
                loParam.Data[i] = Sigmoid(lo.Data[i]);

                this.cSave[batchID].Data[i] = laParam.Data[i] * liParam.Data[i] + lfParam.Data[i] * this.cSave[batchID].Data[i];
                this.hParam[batchID].Data[i] = loParam.Data[i] * Math.Tanh(this.cSave[batchID].Data[i]);
            }

            this.cParam[batchID].Push(new NdArray(this.cSave[batchID]));
            this.aParam[batchID].Push(laParam);
            this.iParam[batchID].Push(liParam);
            this.fParam[batchID].Push(lfParam);
            this.oParam[batchID].Push(loParam);

            return this.hParam[batchID];
        }

        public override void ResetState()
        {
            this.cParam = null;
            this.hParam = new NdArray[1];
        }

        protected override NdArray BackwardSingle(NdArray gh, int batchID = 0)
        {
            if (this.gxPrev[batchID] != null)
            {
                for (int i = 0; i < 4; i++)
                {
                    var ghPre = this.lateral[i].Backward(this.gxPrev[batchID][i], batchID);

                    for (int j = 0; j < ghPre.Length; j++)
                    {
                        gh.Data[j] += ghPre.Data[j];
                    }
                }
            }

            var lcPrev = this.cPrev[batchID].Pop();
            if (this.gcPrev[batchID] == null)
            {
                this.gcPrev[batchID] = NdArray.EmptyLike(gh);
            }

            NdArray ga = NdArray.EmptyLike(gh);
            NdArray gi = NdArray.EmptyLike(gh);
            NdArray gf = NdArray.EmptyLike(gh);
            NdArray go = NdArray.EmptyLike(gh);

            var cPrePara = this.cParam[batchID].Pop();
            var aPrev = this.aParam[batchID].Pop();
            var iPrev = this.iParam[batchID].Pop();
            var fPrev = this.fParam[batchID].Pop();
            var oPrev = this.oParam[batchID].Pop();

            for (int i = 0; i < this.gcPrev[batchID].Length; i++)
            {
                var co = Math.Tanh(cPrePara.Data[i]);

                this.gcPrev[batchID].Data[i] = gh.Data[i] * oPrev.Data[i] * GradTanh(co) + this.gcPrev[batchID].Data[i]; //gcPrevが次回のgc
                ga.Data[i] = this.gcPrev[batchID].Data[i] * iPrev.Data[i] * GradTanh(aPrev.Data[i]);
                gi.Data[i] = this.gcPrev[batchID].Data[i] * aPrev.Data[i] * GradSigmoid(iPrev.Data[i]);
                gf.Data[i] = this.gcPrev[batchID].Data[i] * lcPrev.Data[i] * GradSigmoid(fPrev.Data[i]);
                go.Data[i] = gh.Data[i] * co * GradSigmoid(oPrev.Data[i]);

                this.gcPrev[batchID].Data[i] *= fPrev.Data[i]; //multiply f here
            }

            var r = this.RestoreGates(ga, gi, gf, go);
            this.gxPrev[batchID] = r;

            ga = this.upward[0].Backward(r[0], batchID);
            gi = this.upward[1].Backward(r[1], batchID);
            gf = this.upward[2].Backward(r[2], batchID);
            go = this.upward[3].Backward(r[3], batchID);

            double[] gx = new double[ga.Length];
            for (int i = 0; i < ga.Length; i++)
            {
                gx[i] = ga.Data[i] + gi.Data[i] + gf.Data[i] + go.Data[i];
            }

            return NdArray.FromArray(gx);
        }

        public override void InitBatch(int batchCount)
        {
            for (int i = 0; i < 4; i++)
            {
                this.upward[i].InitBatch(batchCount);
                this.lateral[i].InitBatch(batchCount);
            }

            this.hParam = new NdArray[batchCount];

            this.gcPrev = new NdArray[batchCount];
            this.cSave = new NdArray[batchCount];

            this.cPrev = new Stack<NdArray>[batchCount];
            this.aParam = new Stack<NdArray>[batchCount];
            this.iParam = new Stack<NdArray>[batchCount];
            this.fParam = new Stack<NdArray>[batchCount];
            this.oParam = new Stack<NdArray>[batchCount];
            this.cParam = new Stack<NdArray>[batchCount];

            for (int i = 0; i < this.cPrev.Length; i++)
            {
                this.cPrev[i] = new Stack<NdArray>();
                this.aParam[i] = new Stack<NdArray>();
                this.iParam[i] = new Stack<NdArray>();
                this.fParam[i] = new Stack<NdArray>();
                this.oParam[i] = new Stack<NdArray>();
                this.cParam[i] = new Stack<NdArray>();
            }

            this.gxPrev = new NdArray[batchCount][];
        }

        public NdArray Predict(NdArray input)
        {
            return this.ForwardSingle(input);
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

        NdArray[] RestoreGates(params NdArray[] x)
        {
            NdArray[] result =
            {
                NdArray.Zeros(5),
                NdArray.Zeros(5),
                NdArray.Zeros(5),
                NdArray.Zeros(5)
            };

            double[] r = new double[20];

            for (int i = 0; i < 5; i++)
            {
                r[i * 4 + 0] = x[0].Data[i];
                r[i * 4 + 1] = x[1].Data[i];
                r[i * 4 + 2] = x[2].Data[i];
                r[i * 4 + 3] = x[3].Data[i];
            }

            for (int i = 0; i < 4; i++)
            {
                result[i].Data[0] = r[i * 5 + 0];
                result[i].Data[1] = r[i * 5 + 1];
                result[i].Data[2] = r[i * 5 + 2];
                result[i].Data[3] = r[i * 5 + 3];
                result[i].Data[4] = r[i * 5 + 4];
            }

            return result;
        }

        NdArray[] ExtractGates(params List<double>[] x)
        {
            NdArray[] r =
            {
                NdArray.Zeros(5),
                NdArray.Zeros(5),
                NdArray.Zeros(5),
                NdArray.Zeros(5)
            };

            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < x[i].Count / 4; j++)
                {
                    r[0].Data[j] += x[i][j * 4];
                    r[1].Data[j] += x[i][j * 4 + 1];
                    r[2].Data[j] += x[i][j * 4 + 2];
                    r[3].Data[j] += x[i][j * 4 + 3];
                }
            }

            return r;
        }
    }
}
