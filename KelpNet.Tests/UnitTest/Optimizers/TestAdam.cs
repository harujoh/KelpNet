using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestAdam
    {
        [TestMethod]
        public void AdamRandomTest()
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


            float alpha = (float)Mother.Dice.NextDouble(); //0.001f
            float beta1 = (float)Mother.Dice.NextDouble(); //0.9f;
            float beta2 = (float)Mother.Dice.NextDouble(); //0.999f;
            float eps = (float)Mother.Dice.NextDouble(); //1e-08f;
            float eta = (float)Mother.Dice.NextDouble(); //1.0f;

            //Chainer
            NChainer.Linear<Real> cLinear = new NChainer.Linear<Real>(inputCount, outputCount, false, w, b);
            NChainer.Adam<Real> cAdam = new NChainer.Adam<Real>(alpha, beta1, beta2, eps, eta);
            cAdam.Setup(cLinear);

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cLinear.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();

            cAdam.Update();

            //KelpNet
            KelpNet.CL.Linear<Real> linear = new KelpNet.CL.Linear<Real>(inputCount, outputCount, false, w, b);
            KelpNet.Adam<Real> adam = new Adam<Real>(alpha, beta1, beta2, eps, eta);
            adam.SetUp(linear);

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = linear.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();

            adam.Update();


            Real[] cW = ((Real[,])cLinear.W.Data).Flatten();
            Real[] cb = (Real[])cLinear.b.Data;

            //許容範囲を算出
            Real delta = 0.00001f;

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
