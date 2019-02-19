using KelpNet;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNetTests
{
    [TestClass]
    public class TestLinear
    {
        [TestMethod]
        public void RandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int inputCount = Mother.Dice.Next(2, 50);
            int outputCount = Mother.Dice.Next(2, 50);

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { 1, inputCount });

            Real[,] dummyGy = (Real[,])Initializer.GetRealNdArray(new[] { 1, outputCount });
            Real[,] w = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, inputCount });

            Real[] b = Initializer.GetRealArray(outputCount);

            //Chainer
            NChainer.Linear<Real> cLinear = new NChainer.Linear<Real>(inputCount, outputCount, false, Real.ToBaseNdArray(w), Real.ToBaseArray(b));

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cLinear.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.Linear linear = new KelpNet.Linear(inputCount, outputCount, false, w, b);

            NdArray x = new NdArray(input);

            NdArray y = linear.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,])cY.Data);
            Real[] cXgrad = Real.ToRealArray((Real[,])cX.Grad);

            Real[] cWgrad = Real.ToRealArray((Real[,])cLinear.W.Grad);
            Real[] cbgrad = (Real[])cLinear.b.Grad;

            //許容範囲を算出
            double delta = 0.00001;

            //y
            Assert.AreEqual(cYdata.Length, y.Data.Length);
            for (int i = 0; i < y.Data.Length; i++)
            {
                Assert.AreEqual(cYdata[i], y.Data[i], delta);
            }

            //x.grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }

            //W.grad
            Assert.AreEqual(cWgrad.Length, linear.Weight.Grad.Length);
            for (int i = 0; i < linear.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad[i], linear.Weight.Grad[i], delta);
            }

            //b.grad
            Assert.AreEqual(cbgrad.Length, linear.Bias.Grad.Length);
            for (int i = 0; i < linear.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad[i], linear.Bias.Grad[i], delta);
            }
        }
    }
}
