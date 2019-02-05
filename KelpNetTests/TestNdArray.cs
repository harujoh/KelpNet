using ChainerCore;
using KelpNet.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NConstrictor;

namespace KelpNetTests
{
    [TestClass]
    public class TestNdArray
    {
        [TestMethod]
        public void TestBackward()
        {
            PyMain py = Python.Main;
            Chainer.Initialize();

            //Make random value.
            float val = Mother.Dice.Next();

            //Chainer
            py["x"] = new Variable<float>(new[] { val });
            py["y"] = py["x"] * py["x"] + py["x"] + 1.0f;

            py["y"]["backward"].Call();
            var pyGy = py["x"]["grad_var"]["data"].ToArray<float>();

            //KelpNet
            NdArray x = new NdArray(new[] { val });
            NdArray y = x * x + x + 1.0f;

            y.Backward();
            var gy = Real.ToBaseArray(x.Grad);

            //Check
            CollectionAssert.AreEqual(pyGy, gy);
        }
    }
}
