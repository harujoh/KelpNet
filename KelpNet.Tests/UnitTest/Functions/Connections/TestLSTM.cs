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

            int batchCount = Mother.Dice.Next(1,5);
            int inputCount = Mother.Dice.Next(1, 5);
            int outputCount = Mother.Dice.Next(1, 5);

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, inputCount });
            Real[,] dummyGy = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, outputCount });

            Real[,] upwardInit = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, inputCount });
            Real[,] lateralInit = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, outputCount });
            Real[,,] biasInit = (Real[,,])Initializer.GetRealNdArray(new[] { 1, outputCount, 1 });
            Real[,,] forgetBiasInit = (Real[,,])Initializer.GetRealNdArray(new[] { 1, outputCount, 1 });

            //Chainer
            NChainer.LSTM<Real> clstm = new NChainer.LSTM<Real>(inputCount, outputCount, Real.ToBaseNdArray(lateralInit), Real.ToBaseNdArray(upwardInit), Real.ToBaseNdArray(biasInit), Real.ToBaseNdArray(forgetBiasInit));

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));
            Variable<Real> cY = clstm.Forward(cX);

            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();

            //KelpNet
            KelpNet.LSTM lstm = new KelpNet.LSTM(inputCount, outputCount, lateralInit, upwardInit, biasInit, forgetBiasInit);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { inputCount }, batchCount);
            NdArray y = lstm.Forward(x)[0];

            y.Grad = Real.ToRealArray(dummyGy);

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
            Real[,] input2 = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, inputCount });
            Real[,] dummyGy2 = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, outputCount });

            //Chainer
            Variable<Real> cX2 = new Variable<Real>(Real.ToBaseNdArray(input2));
            Variable<Real> cY2 = clstm.Forward(cX2);

            cY2.Grad = Real.ToBaseNdArray(dummyGy2);

            cY2.Backward();

            //KelpNet
            NdArray x2 = new NdArray(Real.ToRealArray(input2), new[] { inputCount }, batchCount);
            NdArray y2 = lstm.Forward(x2)[0];

            y2.Grad = Real.ToRealArray(dummyGy2);

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
            for (int i = 0; i < clateralWGrad.Length; i++)
            {
                Assert.AreEqual(clateralWGrad[i + wLen * 0], lstm.lateral.Weight.Grad[i], delta);
            }

            //経由が多いため誤差が大きい
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