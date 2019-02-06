using ChainerCore;
using KelpNet.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NConstrictor;

using RealType = System.Single;
//using RealType = System.Double;

namespace KelpNetTests
{
    [TestClass]
    public class TestNdArray
    {
        [TestMethod]
        public void TestBasicMath()
        {
            PyMain py = Python.Main;
            Chainer.Initialize();

            //Make random value.
            float val = Mother.Dice.Next();

            //Chainer
            py["x"] = new Variable<RealType>(new[] { val });

            //Add
            py["add"] = 2 + py["x"] + py["x"] + 2;

            py["add"]["backward"].Call();
            RealType[] pyGadd = (RealType[])py["x"]["grad_var"]["data"].ToArray<RealType>();

            //Mul
            py["mul"] = 2 * py["x"] * py["x"] * 3;

            py["mul"]["backward"].Call();
            RealType[] pyGmul = (RealType[])py["x"]["grad_var"]["data"].ToArray<RealType>();

            //Sub
            py["sub"] = 30 - py["x"] - py["x"] - 2;

            py["sub"]["backward"].Call();
            RealType[] pyGsub = (RealType[])py["x"]["grad_var"]["data"].ToArray<RealType>();

            //Div
            py["div"] = 50 / py["x"] / py["x"] / 2;

            py["div"]["backward"].Call();
            RealType[] pyGdiv = (RealType[])py["x"]["grad_var"]["data"].ToArray<RealType>();


            //KelpNet
            NdArray x = new NdArray(new[] { val });

            //Add
            NdArray add = 2 + x + x + 2;

            add.Backward();
            RealType[] gadd = (RealType[])Real.ToBaseArray<RealType>(x.Grad);

            //mul
            NdArray mul = 2 * x * x * 3;

            mul.Backward();
            RealType[] gmul = (RealType[])Real.ToBaseArray<RealType>(x.Grad);

            //sub
            NdArray sub = 30 - x - x - 2;

            sub.Backward();
            RealType[] gsub = (RealType[])Real.ToBaseArray<RealType>(x.Grad);

            //mul
            NdArray div = 50 / x / x / 2;

            div.Backward();
            RealType[] gdiv = (RealType[])Real.ToBaseArray<RealType>(x.Grad);


            //Check
            CollectionAssert.AreEqual(pyGadd, gadd);
            CollectionAssert.AreEqual(pyGmul, gmul);
            CollectionAssert.AreEqual(pyGsub, gsub);
            CollectionAssert.AreEqual(pyGdiv, gdiv);
        }
    }
}
