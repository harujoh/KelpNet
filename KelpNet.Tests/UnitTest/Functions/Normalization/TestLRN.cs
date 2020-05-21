using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestLRN
    {
        [TestMethod]
        public void LRNRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int n = Mother.Dice.Next(2, 7);
            float k = (float)Mother.Dice.NextDouble() * 3;

            int batchCount = Mother.Dice.Next(1, 5);
            int ch = Mother.Dice.Next(1, 5);
            int width = Mother.Dice.Next(1, 16);
            int height = Mother.Dice.Next(1, 16);

            Real[,,,] input = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, width);
            Real[,,,] dummyGy = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, width);


            //chainer
            LocalResponseNormalization<Real> cLocalResponseNormalization = new LocalResponseNormalization<Real>(n, k);

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cLocalResponseNormalization.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();


            //kelpnet
            LRN<Real> lrn = new LRN<Real>(n, k);

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = lrn.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();


            Real[] cYdata = ((Real[,,,])cY.Data).Flatten();
            Real[] cXgrad = ((Real[,,,])cX.Grad).Flatten();

            //許容範囲を算出
            Real delta = 0.00001f;

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
        }
    }
}
