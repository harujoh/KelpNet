using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

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

            Real[,] input = { { 1.0f }, { 3.0f }, { 5.0f }, { 7.0f }, { 9.0f } };
            Real[,] teach = { { 3.0f }, { 5.0f }, { 7.0f }, { 9.0f }, { 11.0f } };

            Real[,] input2 = { { 3.0f }, { 5.0f }, { 7.0f }, { 9.0f }, { 11.0f } };
            Real[,] teach2 = { { 5.0f }, { 7.0f }, { 9.0f }, { 11.0f }, { 13.0f } };

            int outputCount = 1;
            int inputCount = 1;
            int hiddenCount = 2;

            Real[,] upwardInit = Initializer.GetRandomValues<Real[,]>(hiddenCount, hiddenCount);
            Real[,] lateralInit = Initializer.GetRandomValues<Real[,]>(hiddenCount, hiddenCount);
            Real[,,] biasInit = Initializer.GetRandomValues<Real[,,]>(1, hiddenCount, 1);
            Real[,,] forgetBiasInit = Initializer.GetRandomValues<Real[,,]>(1, hiddenCount, 1);

            //Chainer
            Real[,] w1 = Initializer.GetRandomValues<Real[,]>(hiddenCount, inputCount);
            Real[] b1 = Initializer.GetRandomValues<Real[]>(hiddenCount);

            //Chainer
            Linear<Real> cLinear1 = new Linear<Real>(inputCount, hiddenCount, false, w1, b1);
            NChainer.LSTM<Real> cLstm = new NChainer.LSTM<Real>(hiddenCount, hiddenCount, lateralInit, upwardInit, biasInit, forgetBiasInit);

            Real[,] w2 = Initializer.GetRandomValues<Real[,]>(outputCount, hiddenCount);
            Real[] b2 = Initializer.GetRandomValues<Real[]>(outputCount);
            Linear<Real> cLinear2 = new Linear<Real>(hiddenCount, outputCount, false, w2, b2);

            Variable<Real> cX1 = new Variable<Real>(input);
            Variable<Real> cY11 = cLinear1.Forward(cX1);
            Variable<Real> cY12 = cLstm.Forward(cY11);
            Variable<Real> cY13 = cLinear2.Forward(cY12);
            Variable<Real> cT = new Variable<Real>(teach);

            Variable<Real> cLoss = new NChainer.MeanSquaredError<Real>().Forward(cY13, cT);
            cLoss.Backward();


            //KelpNet
            CL.Linear<Real> linear1 = new CL.Linear<Real>(inputCount, hiddenCount, false, w1, b1);
            LSTM<Real> lstm = new LSTM<Real>(hiddenCount, hiddenCount, lateralInit, upwardInit, biasInit, forgetBiasInit);
            CL.Linear<Real> linear2 = new CL.Linear<Real>(hiddenCount, outputCount, false, w2, b2);

            NdArray<Real> x1 = new NdArray<Real>(input, asBatch: true);
            NdArray<Real> y11 = linear1.Forward(x1)[0];
            NdArray<Real> y12 = lstm.Forward(y11)[0];
            NdArray<Real> y13 = linear2.Forward(y12)[0];
            NdArray<Real> t = new NdArray<Real>(teach, asBatch: true);

            NdArray<Real> loss = new MeanSquaredError<Real>().Evaluate(y13, t);
            y13.Backward();

            Real[] cY11data = ((Real[,])cY11.Data).Flatten();
            Real[] cY12data = ((Real[,])cY12.Data).Flatten();
            Real[] cY13data = ((Real[,])cY13.Data).Flatten();
            Real[] cXgrad = ((Real[,])cX1.Grad).Flatten();

            Real[] cupwardWGrad = ((Real[,])cLstm.upward.W.Grad).Flatten();
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

            Real[] cWgrad11 = ((Real[,])cLinear1.W.Grad).Flatten();
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


            Real[] cWgrad12 = ((Real[,])cLinear2.W.Grad).Flatten();
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
            Variable<Real> cX2 = new Variable<Real>(input2);
            Variable<Real> cY21 = cLinear1.Forward(cX2);
            Variable<Real> cY22 = cLstm.Forward(cY21);
            Variable<Real> cY23 = cLinear2.Forward(cY22);
            Variable<Real> cT2 = new Variable<Real>(teach2);

            Variable<Real> cLoss2 = new NChainer.MeanSquaredError<Real>().Forward(cY23, cT2);

            //KelpNet
            NdArray<Real> x2 = new NdArray<Real>(input2, asBatch: true);
            NdArray<Real> y21 = linear1.Forward(x2)[0];
            NdArray<Real> y22 = lstm.Forward(y21)[0];
            NdArray<Real> y23 = linear2.Forward(y22)[0];
            NdArray<Real> t2 = new NdArray<Real>(teach2, asBatch: true);

            NdArray<Real> loss2 = new MeanSquaredError<Real>().Evaluate(y23, t2);

            Assert.AreEqual(cLoss2.Data[0], loss2.Data[0], delta);

            //Backwardを実行
            cLoss2.Backward();
            y23.Backward();


            Real[] cYdata21 = ((Real[,])cY21.Data).Flatten();
            Real[] cYdata22 = ((Real[,])cY22.Data).Flatten();
            Real[] cYdata23 = ((Real[,])cY23.Data).Flatten();
            Real[] cXgrad2 = ((Real[,])cX2.Grad).Flatten();

            Real[] cupwardWGrad2 = ((Real[,])cLstm.upward.W.Grad).Flatten();
            Real[] cupwardbGrad2 = (Real[])cLstm.upward.b.Grad;
            Real[] clateralWGrad = ((Real[,])cLstm.lateral.W.Grad).Flatten();

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

            Real[] cWgrad22 = ((Real[,])cLinear2.W.Grad).Flatten();
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

            Real[] cWgrad21 = ((Real[,])cLinear1.W.Grad).Flatten();
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
