using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestSGD
    {
        [TestMethod]
        public void SGDRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int inputCount = Mother.Dice.Next(2, 50);
            int outputCount = Mother.Dice.Next(2, 50);
            int batchCount = Mother.Dice.Next(1, 5);

            Real[,] input = Initializer.GetRandomValues<Real[,]>(batchCount, inputCount);

            Real[,] dummyGy = Initializer.GetRandomValues<Real[,]>(batchCount, outputCount);
            Real[,] w = Initializer.GetRandomValues<Real[,]>(outputCount, inputCount);

            Real[] b = Initializer.GetRandomValues<Real[]>(outputCount);

            //Chainer
            Linear<Real> cLinear = new Linear<Real>(inputCount, outputCount, false, w, b);
            NChainer.SGD<Real> cSgd = new NChainer.SGD<Real>();
            cSgd.Setup(cLinear);

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cLinear.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();

            cSgd.Update();

            //KelpNet
            CL.Linear<Real> linear = new CL.Linear<Real>(inputCount, outputCount, false, w, b);
            KelpNet.SGD<Real> sgd = new SGD<Real>();
            sgd.SetUp(linear);

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = linear.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();

            sgd.Update();


            Real[] cW = ((Real[,])cLinear.W.Data).Flatten();
            Real[] cb = (Real[])cLinear.b.Data;

            //許容範囲を算出
            double delta = 0.00001;

            //W.grad
            Assert.AreEqual(cW.Length, linear.Weight.Data.Length);
            for (int i = 0; i < linear.Weight.Data.Length; i++)
            {
                Assert.AreEqual(cW[i], linear.Weight.Data[i], delta);
            }

            //b.grad
            Assert.AreEqual(cb.Length, linear.Bias.Data.Length);
            for (int i = 0; i < linear.Bias.Data.Length; i++)
            {
                Assert.AreEqual(cb[i], linear.Bias.Data[i], delta);
            }
        }
    }
}