//using System.Linq;
//using KelpNet;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
//using NChainer;
//using NConstrictor;

//namespace KelpNetTests
//{
//    [TestClass]
//    public class TestBasicMath
//    {
//        [TestMethod]
//        public void Run()
//        {
//            Python.Initialize();
//            Chainer.Initialize();

//            //Make random value.
//            Real[] val = { Mother.Dice.Next() };

//            //Chainer
//            Variable cX = new Variable((PyArray<float>)Real.ToBaseArray(val));

//            //Add
//            Variable cAdd = Python.GetNamelessObject(2 + cX + cX + 2);

//            cAdd.Backward();
//            Real[] pyGadd = (Real[])((PyArray<Real>)cX.Grad).ToArray();

//            //Mul
//            Variable cMul = Python.GetNamelessObject(2 * cX * cX * 3);

//            cMul.Backward();
//            Real[] pyGmul = (Real[])((PyArray<Real>)cX.Grad).ToArray();

//            //Sub
//            Variable cSub = 30 - cX - cX - 2;

//            cSub.Backward();
//            //Real[] pyGsub = (Real[])((PyArray<Real>)cX.Grad).ToArray();

//            //Div
//            Variable cDiv = 50 / cX / cX / 2;

//            cDiv.Backward();
//            Real[] pyGdiv = (Real[])((PyArray<Real>)cX.Grad).ToArray();


//            //KelpNet
//            NdArray x = new NdArray(val);

//            //Add
//            NdArray add = 2 + x + x + 2;

//            add.Backward();
//            Real[] gadd = x.Grad.ToArray();

//            //mul
//            NdArray mul = 2 * x * x * 3;

//            mul.Backward();
//            Real[] gmul = x.Grad.ToArray();

//            //sub
//            NdArray sub = 30 - x - x - 2;

//            sub.Backward();
//            Real[] gsub = x.Grad.ToArray();

//            //mul
//            NdArray div = 50 / x / x / 2;

//            div.Backward();
//            Real[] gdiv = x.Grad.ToArray();


//            //Check
//            CollectionAssert.AreEqual(pyGadd, gadd);
//            //CollectionAssert.AreEqual(pyGmul, gmul);
//            //CollectionAssert.AreEqual(pyGsub, gsub);
//            CollectionAssert.AreEqual(pyGdiv, gdiv);
//        }
//    }
//}


using System.Linq;
using KelpNet;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNetTests
{
    [TestClass]
    public class TestBasicMath
    {
        [TestMethod]
        public void Run()
        {
            Python py = new Python();
            Chainer.Initialize();

            //Make random value.
            Real[] val = { Mother.Dice.Next() };

            //Chainer
            Variable cX = new Variable((PyArray<float>)Real.ToBaseArray(val));
            py["x"] = cX;

            //Add
            Variable cAdd = 2 + cX + cX + 2;

            cAdd.Backward();
            Real[] pyGadd = (Real[])((PyArray<Real>)py["x"]["grad_var"]["data"]).ToArray();

            //Mul
            Variable cMul = 2 * cX * cX * 3;

            cMul.Backward();
            Real[] pyGmul = (Real[])((PyArray<Real>)py["x"]["grad_var"]["data"]).ToArray();

            //Sub
            Variable cSub = 30 - cX - cX - 2;

            cSub.Backward();
            Real[] pyGsub = (Real[])((PyArray<Real>)py["x"]["grad_var"]["data"]).ToArray();

            //Div
            Variable cDiv = 50 / cX / cX / 2;

            cDiv.Backward();
            Real[] pyGdiv = (Real[])((PyArray<Real>)py["x"]["grad_var"]["data"]).ToArray();


            //KelpNet
            NdArray x = new NdArray(val);

            //Add
            NdArray add = 2 + x + x + 2;

            add.Backward();
            Real[] gadd = x.Grad.ToArray();

            //mul
            NdArray mul = 2 * x * x * 3;

            mul.Backward();
            Real[] gmul = x.Grad.ToArray();

            //sub
            NdArray sub = 30 - x - x - 2;

            sub.Backward();
            Real[] gsub = x.Grad.ToArray();

            //mul
            NdArray div = 50 / x / x / 2;

            div.Backward();
            Real[] gdiv = x.Grad.ToArray();


            //Check
            CollectionAssert.AreEqual(pyGadd, gadd);
            CollectionAssert.AreEqual(pyGmul, gmul);
            CollectionAssert.AreEqual(pyGsub, gsub);
            CollectionAssert.AreEqual(pyGdiv, gdiv);
        }
    }
}