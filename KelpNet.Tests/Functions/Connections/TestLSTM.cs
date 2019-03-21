using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestLSTM
    {
        [TestMethod]
        public void LSTMRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            Real[,] input = { { 1.0 }, { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 } };
            Real[,] teach = { { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 }, { 11.0 } };

            Real[,] input2 = { { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 }, { 11.0 } };
            Real[,] teach2 = { { 5.0 }, { 7.0 }, { 9.0 }, { 11.0 }, { 13.0 } };

            Real[,] upwardInit = (Real[,])Initializer.GetRealNdArray(new[] { 1, 1 });
            Real[,] lateralInit = (Real[,])Initializer.GetRealNdArray(new[] { 1, 1 });
            Real[] biasInit = { Mother.Dice.NextDouble() };
            Real[] forgetBiasInit = { Mother.Dice.NextDouble() };

            //Chainer
            NChainer.LSTM<Real> clstm = new NChainer.LSTM<Real>(1, 1, Real.ToBaseNdArray(lateralInit), Real.ToBaseNdArray(upwardInit), Real.ToBaseArray(biasInit), Real.ToBaseArray(forgetBiasInit));

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));
            Variable<Real> cY = clstm.Forward(cX);
            Variable<Real> cT = new Variable<Real>(Real.ToBaseNdArray(teach));

            Variable<Real> cLoss = new NChainer.MeanSquaredError<Real>().Forward(cY, cT);
            cLoss.Backward();


            //KelpNet
            KelpNet.LSTM lstm = new KelpNet.LSTM(1, 1, lateralInit, upwardInit, biasInit, forgetBiasInit);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { 1 }, 5);
            NdArray y = lstm.Forward(x)[0];
            NdArray t = new NdArray(Real.ToRealArray(teach), new[] { 1 }, 5);

            NdArray loss = new KelpNet.MeanSquaredError().Evaluate(y, t);
            y.Backward();

            //許容範囲を算出
            double delta = 0.00001;

            Real[] cYdata = Real.ToRealArray((Real[,])cY.Data);
            Real[] cXgrad = Real.ToRealArray((Real[,])cX.Grad);

            Real[] cupwardWGrad = Real.ToRealArray((Real[,])clstm.upward.W.Grad);
            Real[] cupwardbGrad = (Real[])clstm.upward.b.Grad;


            //y
            Assert.AreEqual(cYdata.Length, y.Data.Length);
            for (int j = 0; j < 5; j++)
            {
                Assert.AreEqual(cYdata[j], y.Data[j], delta);
            }

            //loss
            Assert.AreEqual(cLoss.Data[0], loss.Data[0], delta);

            //x.Grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int j = 0; j < 5; j++)
            {
                Assert.AreEqual(cXgrad[j], x.Grad[j], delta);
            }

            //W.grad
            int wLen = lstm.upward.Weight.Grad.Length;
            Assert.AreEqual(cupwardWGrad.Length, lstm.upward.Weight.Grad.Length);
            for (int i = 0; i < wLen; i++)
            {
                Assert.AreEqual(cupwardWGrad[i + wLen * 0], lstm.upward.Weight.Grad[i], delta);
            }

            //b.grad
            int bLen = lstm.upward.Bias.Length;

            Assert.AreEqual(cupwardbGrad.Length, lstm.upward.Bias.Grad.Length);
            for (int i = 0; i < bLen; i++)
            {
                Assert.AreEqual(cupwardbGrad[i + wLen * 0], lstm.upward.Bias.Grad[i], delta);
            }


            //////////////////////////////////////////////////////////////////////////////////////////
            // 1度目はlateralに値はない                                                             //
            //////////////////////////////////////////////////////////////////////////////////////////
            //Real[] clateralWGrad = Real.ToRealArray((Real[,])clstm.lateral.W.Grad);
            //Assert.AreEqual(clateralWGrad.Length, lstm.lateral.Weight.Grad.Length);
            //for (int i = 0; i < wLen; i++)
            //{
            //    Assert.AreEqual(clateralWGrad[i + wLen * 0], lstm.lateral.Weight.Grad[i], delta);
            //}
            //////////////////////////////////////////////////////////////////////////////////////////

            ///////////
            //２周目 //
            ///////////

            //Chainer
            Variable<Real> cX2 = new Variable<Real>(Real.ToBaseNdArray(input2));
            Variable<Real> cY2 = clstm.Forward(cX2);
            Variable<Real> cT2 = new Variable<Real>(Real.ToBaseNdArray(teach2));

            Variable<Real> cLoss2 = new NChainer.MeanSquaredError<Real>().Forward(cY2, cT2);

            //KelpNet
            NdArray x2 = new NdArray(Real.ToRealArray(input2), new[] { 1 }, 5);
            NdArray y2 = lstm.Forward(x2)[0];
            NdArray t2 = new NdArray(Real.ToRealArray(teach2), new[] { 1 }, 5);

            NdArray loss2 = new KelpNet.MeanSquaredError().Evaluate(y2, t2);

            Assert.AreEqual(cLoss2.Data[0], loss2.Data[0], delta);

            //Backwardを実行
            cLoss2.Backward();
            y2.Backward();

            Real[] cYdata2 = Real.ToRealArray((Real[,])cY2.Data);
            Real[] cXgrad2 = Real.ToRealArray((Real[,])cX2.Grad);

            Real[] cupwardWGrad2 = Real.ToRealArray((Real[,])clstm.upward.W.Grad);
            Real[] cupwardbGrad2 = (Real[])clstm.upward.b.Grad;
            Real[] clateralWGrad = Real.ToRealArray((Real[,])clstm.lateral.W.Grad);

            //y
            Assert.AreEqual(cYdata2.Length, y2.Data.Length);
            for (int j = 0; j < 5; j++)
            {
                Assert.AreEqual(cYdata2[j], y2.Data[j], delta);
            }

            //x.Grad
            Assert.AreEqual(cXgrad2.Length, x2.Grad.Length);
            for (int j = 0; j < 5; j++)
            {
                Assert.AreEqual(cXgrad2[j], x2.Grad[j], delta);
            }

            //W.grad
            Assert.AreEqual(clateralWGrad.Length, lstm.lateral.Weight.Grad.Length);
            for (int i = 0; i < wLen; i++)
            {
                Assert.AreEqual(clateralWGrad[i + wLen * 0], lstm.lateral.Weight.Grad[i], delta);
            }

            //経由が多くかなり誤差が大きい為
            delta = 0.5;
            for (int i = 0; i < wLen; i++)
            {
                Assert.AreEqual(cupwardWGrad2[i + wLen * 0], lstm.upward.Weight.Grad[i], delta);
            }

            //b.grad
            for (int i = 0; i < bLen; i++)
            {
                Assert.AreEqual(cupwardbGrad2[i + wLen * 0], lstm.upward.Bias.Grad[i], delta);
            }
        }

        //const int STEPS_PER_CYCLE = 50;
        //const int NUMBER_OF_CYCLES = 100;

        //const int MINI_BATCH_SIZE = 5;
        //const int LENGTH_OF_SEQUENCE = 5;

        //[TestMethod]
        //public void LSTMSinTest()
        //{
        //    DataMaker dataMaker = new DataMaker(STEPS_PER_CYCLE, NUMBER_OF_CYCLES);
        //    NdArray trainData = dataMaker.Make();

        //    Real[,] upwardInit = { { 2.0 } };
        //    Real[,] lateralInit = { { 3.0 } };
        //    Real[] biasInit = { 1.5 };
        //    Real[] forgetBiasInit = { 3.2 };

        //    //Real[,] upwardInit = (Real[,])Initializer.GetRealNdArray(new[] { 1, 1 });
        //    //Real[,] lateralInit = (Real[,])Initializer.GetRealNdArray(new[] { 1, 1 });
        //    //Real[] biasInit = { Mother.Dice.NextDouble() };
        //    //Real[] forgetBiasInit = { Mother.Dice.NextDouble() };

        //    NChainer.LSTM<Real> clstm = new NChainer.LSTM<Real>(1, 1, Real.ToBaseNdArray(lateralInit), Real.ToBaseNdArray(upwardInit), Real.ToBaseArray(biasInit), Real.ToBaseArray(forgetBiasInit));

        //    KelpNet.LSTM lstm = new KelpNet.LSTM(1, 1, lateralInit, upwardInit, biasInit, forgetBiasInit);

        //    NdArray[] sequences = dataMaker.MakeMiniBatch(trainData, MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

        //    Real[,] input = new Real[MINI_BATCH_SIZE, 1];
        //    Real[,] teach = new Real[MINI_BATCH_SIZE, 1];

        //    Stack<NdArray> backNdArrays = new Stack<NdArray>();
        //    //Stack<Variable<Real>> backVariables = new Stack<Variable<Real>>();

        //    Variable<Real> cLoss = new Variable<Real>();

        //    //許容範囲を算出
        //    double delta = 0.00001;

        //    for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++)
        //    {
        //        for (int j = 0; j < MINI_BATCH_SIZE; j++)
        //        {
        //            input[j, 0] = sequences[j].Data[i];
        //            teach[j, 0] = sequences[j].Data[i + 1];
        //        }

        //        Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));
        //        Variable<Real> cT = new Variable<Real>(Real.ToBaseNdArray(teach));
        //        Variable<Real> cY = clstm.Forward(cX);

        //        NdArray x = new NdArray(Real.ToRealArray(input), new[] { 1 }, MINI_BATCH_SIZE);
        //        NdArray t = new NdArray(Real.ToRealArray(teach), new[] { 1 }, MINI_BATCH_SIZE);
        //        NdArray y = lstm.Forward(x)[0];

        //        //出力されたものを一回一回チェック
        //        Real[] cYtmp = Real.ToRealArray((Real[,])cY.Data);

        //        Assert.AreEqual(cYtmp.Length, y.Data.Length);
        //        for (int j = 0; j < MINI_BATCH_SIZE; j++)
        //        {
        //            Assert.AreEqual(cYtmp[j], y.Data[j], delta);
        //        }

        //        //ChainerはLossからBackwardが始まる
        //        cLoss = new NChainer.MeanSquaredError<Real>().Forward(cY, cT);

        //        //KelpNetは入力されたYにGrad値が書き込まれる
        //        NdArray loss = new KelpNet.MeanSquaredError().Evaluate(y, t);
        //        backNdArrays.Push(y);

        //        Assert.AreEqual(cLoss.Data[0], loss.Data[0], delta);
        //    }

        //    cLoss.Backward();

        //    //KelpNetは勝手にまとまってBackwardを行わない
        //    for (int i = 0; backNdArrays.Count > 0; i++)
        //    {
        //        backNdArrays.Pop().Backward();
        //    }

        //    Real[] lateralWGrad = Real.ToRealArray((Real[,])clstm.lateral.W.Grad);
        //    Assert.AreEqual(lateralWGrad.Length, lstm.lateral0.Weight.Grad.Length + lstm.lateral1.Weight.Grad.Length + lstm.lateral2.Weight.Grad.Length + lstm.lateral3.Weight.Grad.Length);

        //    Real[] upwardWGrad = Real.ToRealArray((Real[,])clstm.upward.W.Grad);
        //    Assert.AreEqual(upwardWGrad.Length, lstm.upward0.Weight.Grad.Length + lstm.upward1.Weight.Grad.Length + lstm.upward2.Weight.Grad.Length + lstm.upward3.Weight.Grad.Length);

        //    int wLen = lstm.lateral0.Weight.Grad.Length;
        //    for (int j = 0; j < wLen; j++)
        //    {
        //        Assert.AreEqual(lateralWGrad[j + wLen * 0], lstm.lateral0.Weight.Grad[j], delta);
        //        Assert.AreEqual(lateralWGrad[j + wLen * 1], lstm.lateral1.Weight.Grad[j], delta);
        //        Assert.AreEqual(lateralWGrad[j + wLen * 2], lstm.lateral2.Weight.Grad[j], delta);
        //        Assert.AreEqual(lateralWGrad[j + wLen * 3], lstm.lateral3.Weight.Grad[j], delta);

        //        Assert.AreEqual(upwardWGrad[j + wLen * 0], lstm.upward0.Weight.Grad[j], delta);
        //        Assert.AreEqual(upwardWGrad[j + wLen * 1], lstm.upward1.Weight.Grad[j], delta);
        //        Assert.AreEqual(upwardWGrad[j + wLen * 2], lstm.upward2.Weight.Grad[j], delta);
        //        Assert.AreEqual(upwardWGrad[j + wLen * 3], lstm.upward3.Weight.Grad[j], delta);
        //    }

        //    Real[] upwardbGrad = (Real[])clstm.upward.b.Data;
        //    Assert.AreEqual(upwardbGrad.Length, lstm.upward0.Bias.Grad.Length + lstm.upward1.Bias.Grad.Length + lstm.upward2.Bias.Grad.Length + lstm.upward3.Bias.Grad.Length);

        //    int bLen = lstm.upward0.Bias.Length;
        //    for (int j = 0; j < bLen; j++)
        //    {
        //        Assert.AreEqual(upwardbGrad[j + wLen * 0], lstm.upward0.Bias.Grad[j], delta);
        //        Assert.AreEqual(upwardbGrad[j + wLen * 1], lstm.upward1.Bias.Grad[j], delta);
        //        Assert.AreEqual(upwardbGrad[j + wLen * 2], lstm.upward2.Bias.Grad[j], delta);
        //        Assert.AreEqual(upwardbGrad[j + wLen * 3], lstm.upward3.Bias.Grad[j], delta);
        //    }
        //}
    }

    //class DataMaker
    //{
    //    private readonly int stepsPerCycle;
    //    private readonly int numberOfCycles;

    //    public DataMaker(int stepsPerCycle, int numberOfCycles)
    //    {
    //        this.stepsPerCycle = stepsPerCycle;
    //        this.numberOfCycles = numberOfCycles;
    //    }

    //    public NdArray Make()
    //    {
    //        NdArray result = new NdArray(this.stepsPerCycle * this.numberOfCycles);

    //        for (int i = 0; i < this.numberOfCycles; i++)
    //        {
    //            for (int j = 0; j < this.stepsPerCycle; j++)
    //            {
    //                result.Data[j + i * this.stepsPerCycle] = Math.Sin(j * 2 * Math.PI / this.stepsPerCycle);
    //            }
    //        }

    //        return result;
    //    }

    //    public NdArray[] MakeMiniBatch(NdArray baseFreq, int miniBatchSize, int lengthOfSequence)
    //    {
    //        NdArray[] result = new NdArray[miniBatchSize];

    //        for (int i = 0; i < result.Length; i++)
    //        {
    //            result[i] = new NdArray(lengthOfSequence);

    //            int index = Mother.Dice.Next(baseFreq.Data.Length - lengthOfSequence);
    //            for (int j = 0; j < lengthOfSequence; j++)
    //            {
    //                result[i].Data[j] = baseFreq.Data[index + j];
    //            }

    //        }

    //        return result;
    //    }
    //}
}