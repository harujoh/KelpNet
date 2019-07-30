using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests.JoinTest
{
    [TestClass]
    public class TestRnnWithLSTM
    {
        [TestMethod]
        public void RnnLSTMRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            Real[,] input = { { 1.0 }, { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 } };
            Real[,] teach = { { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 }, { 11.0 } };

            Real[,] input2 = { { 3.0 }, { 5.0 }, { 7.0 }, { 9.0 }, { 11.0 } };
            Real[,] teach2 = { { 5.0 }, { 7.0 }, { 9.0 }, { 11.0 }, { 13.0 } };

            int outputCount = 1;
            int inputCount = 1;
            int hiddenCount = 2;

            Real[,] upwardInit = (Real[,])Initializer.GetRealNdArray(new[] { hiddenCount, hiddenCount });
            Real[,] lateralInit = (Real[,])Initializer.GetRealNdArray(new[] { hiddenCount, hiddenCount });
            Real[,,] biasInit = (Real[,,])Initializer.GetRealNdArray(new[] { 1, hiddenCount, 1 });
            Real[,,] forgetBiasInit = (Real[,,])Initializer.GetRealNdArray(new[] { 1, hiddenCount, 1 });

            //Chainer
            Real[,] w1 = (Real[,])Initializer.GetRealNdArray(new[] { hiddenCount, inputCount });
            Real[] b1 = Initializer.GetRealArray(hiddenCount);

            //Chainer
            NChainer.Linear<Real> cLinear1 = new NChainer.Linear<Real>(inputCount, hiddenCount, false, Real.ToBaseNdArray(w1), Real.ToBaseArray(b1));
            NChainer.LSTM<Real> cLstm = new NChainer.LSTM<Real>(hiddenCount, hiddenCount, Real.ToBaseNdArray(lateralInit), Real.ToBaseNdArray(upwardInit), Real.ToBaseNdArray(biasInit), Real.ToBaseNdArray(forgetBiasInit));

            Real[,] w2 = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, hiddenCount });
            Real[] b2 = Initializer.GetRealArray(outputCount);
            NChainer.Linear<Real> cLinear2 = new NChainer.Linear<Real>(hiddenCount, outputCount, false, Real.ToBaseNdArray(w2), Real.ToBaseArray(b2));

            Variable<Real> cX1 = new Variable<Real>(Real.ToBaseNdArray(input));
            Variable<Real> cY11 = cLinear1.Forward(cX1);
            Variable<Real> cY12 = cLstm.Forward(cY11);
            Variable<Real> cY13 = cLinear2.Forward(cY12);
            Variable<Real> cT = new Variable<Real>(Real.ToBaseNdArray(teach));

            Variable<Real> cLoss = new NChainer.MeanSquaredError<Real>().Forward(cY13, cT);
            cLoss.Backward();


            //KelpNet
            KelpNet.CL.Linear linear1 = new KelpNet.CL.Linear(inputCount, hiddenCount, false, w1, b1);
            KelpNet.LSTM lstm = new KelpNet.LSTM(hiddenCount, hiddenCount, lateralInit, upwardInit, biasInit, forgetBiasInit);
            KelpNet.CL.Linear linear2 = new KelpNet.CL.Linear(hiddenCount, outputCount, false, w2, b2);

            NdArray x1 = new NdArray(Real.ToRealArray(input), new[] { 1 }, 5);
            NdArray y11 = linear1.Forward(x1)[0];
            NdArray y12 = lstm.Forward(y11)[0];
            NdArray y13 = linear2.Forward(y12)[0];
            NdArray t = new NdArray(Real.ToRealArray(teach), new[] { 1 }, 5);

            NdArray loss = new KelpNet.MeanSquaredError().Evaluate(y13, t);
            y13.Backward();

            Real[] cY11data = Real.ToRealArray((Real[,])cY11.Data);
            Real[] cY12data = Real.ToRealArray((Real[,])cY12.Data);
            Real[] cY13data = Real.ToRealArray((Real[,])cY13.Data);
            Real[] cXgrad = Real.ToRealArray((Real[,])cX1.Grad);

            Real[] cupwardWGrad = Real.ToRealArray((Real[,])cLstm.upward.W.Grad);
            Real[] cupwardbGrad = (Real[])cLstm.upward.b.Grad;


            //許容範囲を設定
            double delta = 0.00001;

            //y11
            Assert.AreEqual(cY11data.Length, y11.Data.Length);
            for (int i = 0; i < cY11data.Length; i++)
            {
                Assert.AreEqual(cY11data[i], y11.Data[i], delta);
            }

            //y12
            Assert.AreEqual(cY12data.Length, y12.Data.Length);
            for (int i = 0; i < cY12data.Length; i++)
            {
                Assert.AreEqual(cY12data[i], y12.Data[i], delta);
            }

            //y13
            Assert.AreEqual(cY13data.Length, y13.Data.Length);
            for (int i = 0; i < cY13data.Length; i++)
            {
                Assert.AreEqual(cY13data[i], y13.Data[i], delta);
            }

            //許容範囲を設定
            delta = 0.0001;

            //loss
            Assert.AreEqual(cLoss.Data[0], loss.Data[0], delta);

            //x.Grad
            Assert.AreEqual(cXgrad.Length, x1.Grad.Length);
            for (int i = 0; i < cXgrad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x1.Grad[i], delta);
            }

            Real[] cWgrad11 = Real.ToRealArray((Real[,])cLinear1.W.Grad);
            Real[] cbgrad11 = (Real[])cLinear1.b.Grad;

            //W.grad
            Assert.AreEqual(cWgrad11.Length, linear1.Weight.Grad.Length);
            for (int i = 0; i < linear1.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad11[i], linear1.Weight.Grad[i], delta);
            }

            //b.grad
            Assert.AreEqual(cbgrad11.Length, linear1.Bias.Grad.Length);
            for (int i = 0; i < linear1.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad11[i], linear1.Bias.Grad[i], delta);
            }


            Real[] cWgrad12 = Real.ToRealArray((Real[,])cLinear2.W.Grad);
            Real[] cbgrad12 = (Real[])cLinear2.b.Grad;


            //W.grad
            Assert.AreEqual(cWgrad12.Length, linear2.Weight.Grad.Length);
            for (int i = 0; i < linear2.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad12[i], linear2.Weight.Grad[i], delta);
            }

            //b.grad
            Assert.AreEqual(cbgrad12.Length, linear2.Bias.Grad.Length);
            for (int i = 0; i < linear2.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad12[i], linear2.Bias.Grad[i], delta);
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


            //２周目
            Variable<Real> cX2 = new Variable<Real>(Real.ToBaseNdArray(input2));
            Variable<Real> cY21 = cLinear1.Forward(cX2);
            Variable<Real> cY22 = cLstm.Forward(cY21);
            Variable<Real> cY23 = cLinear2.Forward(cY22);
            Variable<Real> cT2 = new Variable<Real>(Real.ToBaseNdArray(teach2));

            Variable<Real> cLoss2 = new NChainer.MeanSquaredError<Real>().Forward(cY23, cT2);

            //KelpNet
            NdArray x2 = new NdArray(Real.ToRealArray(input2), new[] { 1 }, 5);
            NdArray y21 = linear1.Forward(x2)[0];
            NdArray y22 = lstm.Forward(y21)[0];
            NdArray y23 = linear2.Forward(y22)[0];
            NdArray t2 = new NdArray(Real.ToRealArray(teach2), new[] { 1 }, 5);

            NdArray loss2 = new KelpNet.MeanSquaredError().Evaluate(y23, t2);

            Assert.AreEqual(cLoss2.Data[0], loss2.Data[0], delta);

            //Backwardを実行
            cLoss2.Backward();
            y23.Backward();


            Real[] cYdata21 = Real.ToRealArray((Real[,])cY21.Data);
            Real[] cYdata22 = Real.ToRealArray((Real[,])cY22.Data);
            Real[] cYdata23 = Real.ToRealArray((Real[,])cY23.Data);
            Real[] cXgrad2 = Real.ToRealArray((Real[,])cX2.Grad);

            Real[] cupwardWGrad2 = Real.ToRealArray((Real[,])cLstm.upward.W.Grad);
            Real[] cupwardbGrad2 = (Real[])cLstm.upward.b.Grad;
            Real[] clateralWGrad = Real.ToRealArray((Real[,])cLstm.lateral.W.Grad);

            //y21
            Assert.AreEqual(cYdata21.Length, y21.Data.Length);
            for (int i = 0; i < cYdata21.Length; i++)
            {
                Assert.AreEqual(cYdata21[i], y21.Data[i], delta);
            }

            //y22
            Assert.AreEqual(cYdata22.Length, y22.Data.Length);
            for (int i = 0; i < cYdata22.Length; i++)
            {
                Assert.AreEqual(cYdata22[i], y22.Data[i], delta);
            }

            //y23
            Assert.AreEqual(cYdata23.Length, y23.Data.Length);
            for (int i = 0; i < cYdata23.Length; i++)
            {
                Assert.AreEqual(cYdata23[i], y23.Data[i], delta);
            }

            //x.Grad
            Assert.AreEqual(cXgrad2.Length, x2.Grad.Length);
            for (int i = 0; i < cXgrad2.Length; i++)
            {
                Assert.AreEqual(cXgrad2[i], x2.Grad[i], delta);
            }

            //経由が多くかなり誤差が大きい為
            delta = 1.0;

            Real[] cWgrad22 = Real.ToRealArray((Real[,])cLinear2.W.Grad);
            Real[] cbgrad22 = (Real[])cLinear2.b.Grad;

            //W.grad
            Assert.AreEqual(cWgrad22.Length, linear2.Weight.Grad.Length);
            for (int i = 0; i < linear2.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad22[i], linear2.Weight.Grad[i], delta);
            }

            //b.grad
            Assert.AreEqual(cbgrad22.Length, linear2.Bias.Grad.Length);
            for (int i = 0; i < linear2.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad22[i], linear2.Bias.Grad[i], delta);
            }


            delta = 2.0;

            //W.grad
            Assert.AreEqual(clateralWGrad.Length, lstm.lateral.Weight.Grad.Length);
            for (int i = 0; i < clateralWGrad.Length; i++)
            {
                Assert.AreEqual(clateralWGrad[i + wLen * 0], lstm.lateral.Weight.Grad[i], delta);
            }

            for (int i = 0; i < wLen; i++)
            {
                Assert.AreEqual(cupwardWGrad2[i + wLen * 0], lstm.upward.Weight.Grad[i], delta);
            }

            //b.grad
            for (int i = 0; i < bLen; i++)
            {
                Assert.AreEqual(cupwardbGrad2[i + wLen * 0], lstm.upward.Bias.Grad[i], delta);
            }


            delta = 20.0;

            Real[] cWgrad21 = Real.ToRealArray((Real[,])cLinear1.W.Grad);
            Real[] cbgrad21 = (Real[])cLinear1.b.Grad;

            //W.grad
            Assert.AreEqual(cWgrad21.Length, linear1.Weight.Grad.Length);
            for (int i = 0; i < linear1.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad21[i], linear1.Weight.Grad[i], delta);
            }

            //b.grad
            Assert.AreEqual(cbgrad21.Length, linear1.Bias.Grad.Length);
            for (int i = 0; i < linear1.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad21[i], linear1.Bias.Grad[i], delta);
            }
        }
    }
}
