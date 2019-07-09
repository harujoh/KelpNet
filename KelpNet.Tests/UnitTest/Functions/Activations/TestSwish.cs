using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestSwish
    {
        [TestMethod]
        public void SwishRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int ioCount = Mother.Dice.Next(1, 50);
            int batchCount = Mother.Dice.Next(1, 5);

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, ioCount });

            Real[,] dummyGy = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, ioCount });

            Real beta = Mother.Dice.NextDouble();

            //Chainer
            NChainer.Swish<Real> cSwish = new NChainer.Swish<Real>(new[] { ioCount }, beta);

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cSwish.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.Swish swish = new KelpNet.Swish(new[] { ioCount }, beta);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { ioCount }, batchCount);

            NdArray y = swish.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,])cY.Data);
            Real[] cXgrad = Real.ToRealArray((Real[,])cX.Grad);

            Real[] cbgrad = (Real[])cSwish.beta.Grad;

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

            //b.grad
            Assert.AreEqual(cbgrad.Length, swish.Beta.Grad.Length);
            for (int i = 0; i < swish.Beta.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad[i], swish.Beta.Grad[i], delta);
            }
        }
    }
}
