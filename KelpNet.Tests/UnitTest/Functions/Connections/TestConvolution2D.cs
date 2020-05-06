using System;
using KelpNet.CL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

#if DOUBLE
using KelpMath = System.Math;
#elif NETCOREAPP2_0
using KelpMath = System.MathF;
#else
using KelpMath = KelpNet.MathF;
#endif

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestConvolution2D
    {
        [TestMethod]
        public void Conv2DCPURandomTest()
        {
            RandomTest(false);
        }

        [TestMethod]
        public void Conv2DGPURandomTest()
        {
            OpenCL.Initialize();

            if (OpenCL.Enable)
            {
                RandomTest(true);
            }
            else
            {
                Assert.Inconclusive();
            }
        }

        public void RandomTest(bool gpuEnable)
        {
            Python.Initialize();
            Chainer.Initialize();

            int batchCount = Mother.Dice.Next(1, 5);
            int inChCount = Mother.Dice.Next(1, 5);
            int outChCount = Mother.Dice.Next(1, 5);
            int wideSize = Mother.Dice.Next(8, 32);
            int heightSize = Mother.Dice.Next(8, 32);

            int kWidth = Mother.Dice.Next(1, 5);
            int kHeight = Mother.Dice.Next(1, 5);

            int strideX = Mother.Dice.Next(1, 5);
            int strideY = Mother.Dice.Next(1, 5);

            int padX = Mother.Dice.Next(0, 5);
            int padY = Mother.Dice.Next(0, 5);

            int outputHeight = (int)KelpMath.Floor((heightSize - kHeight + padY * 2.0f) / strideY) + 1;
            int outputWidth = (int)KelpMath.Floor((wideSize - kWidth + padX * 2.0f) / strideX) + 1;

            Real[,,,] input = Initializer.GetRandomValues<Real[,,,]>(batchCount, inChCount, heightSize, wideSize);

            Real[,,,] dummyGy = Initializer.GetRandomValues<Real[,,,]>(batchCount, outChCount, outputHeight, outputWidth);
            Real[,,,] w = Initializer.GetRandomValues<Real[,,,]>(outChCount, inChCount, kHeight, kWidth);

            Real[] b = Initializer.GetRandomValues<Real[]>(outChCount);

            //Chainer
            NChainer.Convolution2D<Real> cConvolotion2D = new NChainer.Convolution2D<Real>(
                inChCount, outChCount,
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX },
                false,
                w,
                b);

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cConvolotion2D.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();


            //KelpNet
            CL.Convolution2D<Real> convolution2D = new CL.Convolution2D<Real>(
                inChCount, outChCount,
                new[] { kWidth, kHeight },
                new[] { strideX, strideY },
                new[] { padX, padY },
                false, w, b, gpuEnable: gpuEnable);

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = convolution2D.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();


            Real[] cYdata = ((Real[,,,])cY.Data.Copy()).Flatten();
            Real[] cXgrad = ((Real[,,,])cX.Grad.Copy()).Flatten();

            Real[] cWgrad = ((Real[,,,])cConvolotion2D.W.Grad).Flatten();
            Real[] cbgrad = (Real[])cConvolotion2D.b.Grad;

            //許容範囲を算出
            double delta = 0.00001;

            Assert.AreEqual(cYdata.Length, y.Data.Length);
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            Assert.AreEqual(cWgrad.Length, convolution2D.Weight.Grad.Length);
            Assert.AreEqual(cbgrad.Length, convolution2D.Bias.Grad.Length);

            //y
            for (int i = 0; i < y.Data.Length; i++)
            {
                Assert.AreEqual(cYdata[i], y.Data[i], delta);
            }

            //x.grad
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }

            delta = 0.1;
            //W.grad
            for (int i = 0; i < convolution2D.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad[i], convolution2D.Weight.Grad[i], delta);
            }

            //b.grad
            for (int i = 0; i < convolution2D.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad[i], convolution2D.Bias.Grad[i], delta);
            }
        }
    }
}
