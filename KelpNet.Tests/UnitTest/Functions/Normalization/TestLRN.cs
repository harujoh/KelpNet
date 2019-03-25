using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

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

            Real[,,,] input = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, width });
            Real[,,,] dummyGy = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, ch, height, width });


            //chainer
            NChainer.LocalResponseNormalization<Real> cLocalResponseNormalization = new NChainer.LocalResponseNormalization<Real>(n, k);

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cLocalResponseNormalization.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //kelpnet
            KelpNet.LRN lrn = new LRN(n, k);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { ch, height, width }, batchCount);

            NdArray y = lrn.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,,,])cY.Data);
            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad);

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
        }
    }
}
