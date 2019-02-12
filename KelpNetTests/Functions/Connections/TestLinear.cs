using System.Runtime.CompilerServices;
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
        public void Run()
        {
            Python.Initialize();
            Chainer.Initialize();

            int inputCount = 1 + Mother.Dice.Next() % 49;
            int outputCount = 1 + Mother.Dice.Next() % 49;

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { 1, inputCount });

            Real[,] dummyGy = (Real[,])Initializer.GetRealNdArray(new[] { 1, outputCount });
            Real[,] w = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, inputCount });

            Real[] b = Initializer.GetRealArray(outputCount);

            //Chainer
            NChainer.Linear cLinear = new NChainer.Linear(inputCount, outputCount, false, w, b);

            Variable<Real> cX = new Variable<Real>(Real.ToBaseArray(input));

            Variable<Real> cY = cLinear.Forward(cX);
            cY.Grad = Real.ToBaseArray(dummyGy);

            cY.Backward();

            //KelpNet
            KelpNet.Linear linear = new KelpNet.Linear(inputCount, outputCount, false, w, b);

            NdArray x = new NdArray(input);
            NdArray y = linear.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();

            Real[,] cYdata = (Real[,])cY.Data;
            Real[,] cXgrad = (Real[,])cX.Grad;

            CollectionAssert.AreEqual(Unsafe.As<Real[,], Real[]>(ref cYdata), y.Data);
            CollectionAssert.AreEqual(Unsafe.As<Real[,], Real[]>(ref cXgrad), x.Grad);
        }
    }
}
