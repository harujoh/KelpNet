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

            Real realTypeChecker = 1;

            if (realTypeChecker.GetType() == typeof(float))
            {
                //Chainer
                py["x"] = new Variable<float>(new[] { val });
                py["y"] = py["x"] * py["x"] + py["x"] + 1.0f;

                py["y"]["backward"].Call();
                float[] pyGy = (float[])py["x"]["grad_var"]["data"].ToArray<float>();

                //KelpNet
                NdArray x = new NdArray(new[] { val });
                NdArray y = x * x + x + 1.0f;

                y.Backward();
                float[] gy = (float[])Real.ToBaseArray<float>(x.Grad);

                //Check
                CollectionAssert.AreEqual(pyGy, gy);
            }
            else
            {
                //Chainer
                py["x"] = new Variable<double>(new[] { val });
                py["y"] = py["x"] * py["x"] + py["x"] + 1.0;

                py["y"]["backward"].Call();
                double[] pyGy = (double[])py["x"]["grad_var"]["data"].ToArray<double>();

                //KelpNet
                NdArray x = new NdArray(new[] { val });
                NdArray y = x * x + x + 1.0f;

                y.Backward();
                double[] gy = (double[])Real.ToBaseArray<double>(x.Grad);

                //Check
                CollectionAssert.AreEqual(pyGy, gy);
            }
        }
    }
}
