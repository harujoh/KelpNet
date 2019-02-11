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

            int inputCount = 1 + Mother.Dice.Next() % 99;
            int outputCount = 1 + Mother.Dice.Next() % 99;

            Real[,] input = (Real[,])Initializer.GetRealNdArray(new[] { 1, inputCount });

            Real[] dummyGy = Initializer.GetRealArray(outputCount);
            Real[,] w = (Real[,])Initializer.GetRealNdArray(new[] { outputCount, inputCount });

            Real[] b = Initializer.GetRealArray(outputCount);

            //Chainer
            NChainer.Linear cLinear = new NChainer.Linear(inputCount, outputCount, false, (PyArray<float>)Real.ToBaseArray(w), (PyArray<float>)Real.ToBaseArray(b));

            Variable cX = new Variable((PyArray<float>)Real.ToBaseArray(input));

            Variable cY = cLinear.Forward(cX);

            //KelpNet
            KelpNet.Linear linear = new KelpNet.Linear(inputCount, outputCount, false, w, b);

            NdArray x = new NdArray(input);
            NdArray y = linear.Forward(x)[0];
            y.Grad = dummyGy;

            y.Backward();
        }
    }
}
