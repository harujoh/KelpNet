﻿using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestBasicMath
    {
        [TestMethod]
        public void BasicMathRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            //Make random value.
            Real[] val = { 1 + Mother.Dice.Next() };

            //Chainer
            Variable<Real> cX = new Variable<Real>(val);

            //Add
            Variable<Real> cAdd = 2 + cX + cX + 2;

            cAdd.Backward();
            Real[] pyGadd = (Real[])cX.Grad;

            //Mul
            Variable<Real> cMul = 2 * cX * cX * 3;

            cMul.Backward();
            Real[] pyGmul = (Real[])cX.Grad;

            //Sub
            Variable<Real> cSub = 30 - cX - cX - 2;

            cSub.Backward();
            Real[] pyGsub = (Real[])cX.Grad;

            //Div
            Variable<Real> cDiv = 50 / cX / cX / 2;

            cDiv.Backward();
            Real[] pyGdiv = (Real[])cX.Grad;


            //KelpNet
            NdArray<Real> x = new NdArray<Real>(val);

            //Add
            NdArray<Real> add = 2 + x + x + 2;

            add.Backward();
            Real[] gadd = x.Grad.ToArray(); //このToArrayはコピーのため

            //mul
            NdArray<Real> mul = 2 * x * x * 3;

            mul.Backward();
            Real[] gmul = x.Grad.ToArray();

            //sub
            NdArray<Real> sub = 30 - x - x - 2;

            sub.Backward();
            Real[] gsub = x.Grad.ToArray();

            //mul
            NdArray<Real> div = 50 / x / x / 2;

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