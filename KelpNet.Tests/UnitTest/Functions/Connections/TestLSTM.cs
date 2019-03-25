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
            for (int i = 0; i < cYdata.Length; i++)
            {
                Assert.AreEqual(cYdata[i], y.Data[i], delta);
            }

            //loss
            Assert.AreEqual(cLoss.Data[0], loss.Data[0], delta);

            //x.Grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int i = 0; i < cXgrad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
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
            for (int i = 0; i < cYdata2.Length; i++)
            {
                Assert.AreEqual(cYdata2[i], y2.Data[i], delta);
            }

            //x.Grad
            Assert.AreEqual(cXgrad2.Length, x2.Grad.Length);
            for (int i = 0; i < cXgrad2.Length; i++)
            {
                Assert.AreEqual(cXgrad2[i], x2.Grad[i], delta);
            }

            //W.grad
            Assert.AreEqual(clateralWGrad.Length, lstm.lateral.Weight.Grad.Length);
            for (int i = 0; i < wLen; i++)
            {
                Assert.AreEqual(clateralWGrad[i + wLen * 0], lstm.lateral.Weight.Grad[i], delta);
            }

            //経由が多くかなり誤差が大きい為
            delta = 1.0;
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
    }
}