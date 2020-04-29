using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

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

            Real[,] input = Initializer.GetRandomValues<Real[,]>(batchCount, ioCount);

            Real[,] dummyGy = Initializer.GetRandomValues<Real[,]>(batchCount, ioCount);

            Real beta = (Real)Mother.Dice.NextDouble();

            //Chainer
            NChainer.Swish<Real> cSwish = new NChainer.Swish<Real>(new[] { ioCount }, beta);

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cSwish.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();


            //KelpNet
            Swish<Real> swish = new Swish<Real>(new[] { ioCount }, beta);

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = swish.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();


            Real[] cYdata = ((Real[,])cY.Data).Flatten();
            Real[] cXgrad = ((Real[,])cX.Grad).Flatten();

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
