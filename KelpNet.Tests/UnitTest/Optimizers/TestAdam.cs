using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

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

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, inputCount });
            Real[,] dummyGy = (Real[,])Initializer.GetRealNdArray(new[] { batchCount, outputCount });
            Real[,] w = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, inputCount });
            Real[] b = Initializer.GetRealArray(outputCount);


            float alpha = (float)Mother.Dice.NextDouble(); //0.001f
            float beta1 = (float)Mother.Dice.NextDouble(); //0.9f;
            float beta2 = (float)Mother.Dice.NextDouble(); //0.999f;
            float eps = (float)Mother.Dice.NextDouble(); //1e-08f;
            float eta = (float)Mother.Dice.NextDouble(); //1.0f;

            //Chainer
            NChainer.Linear<Real> cLinear = new NChainer.Linear<Real>(inputCount, outputCount, false, Real.ToBaseNdArray(w), Real.ToBaseArray(b));
            NChainer.Adam<Real> cAdam = new NChainer.Adam<Real>(alpha, beta1, beta2, eps, eta);
            cAdam.Setup(cLinear);

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cLinear.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();

            cAdam.Update();

            //KelpNet
            KelpNet.CL.Linear linear = new KelpNet.CL.Linear(inputCount, outputCount, false, w, b);
            KelpNet.Adam adam = new Adam(alpha, beta1, beta2, eps, eta);
            adam.SetUp(linear);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { inputCount }, batchCount);

            NdArray y = linear.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();

            adam.Update();


            Real[] cW = Real.ToRealArray((Real[,])cLinear.W.Data);
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
