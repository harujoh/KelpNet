using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
using KelpNet.CL;

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

            int outputHeight = (int)Math.Floor((heightSize - kHeight + padY * 2.0) / strideY) + 1;
            int outputWidth = (int)Math.Floor((wideSize - kWidth + padX * 2.0) / strideX) + 1;

            Real[,,,] input = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, inChCount, heightSize, wideSize });

            Real[,,,] dummyGy = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, outChCount, outputHeight, outputWidth });
            Real[,,,] w = (Real[,,,])Initializer.GetRealNdArray(new[] { outChCount, inChCount, kHeight, kWidth });

            Real[] b = Initializer.GetRealArray(outChCount);

            //Chainer
            NChainer.Convolution2D<Real> cConvolotion2D = new NChainer.Convolution2D<Real>(
                inChCount, outChCount,
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX },
                false,
                Real.ToBaseNdArray(w),
                Real.ToBaseArray(b));

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cConvolotion2D.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.CL.Convolution2D convolution2D = new KelpNet.CL.Convolution2D(
                inChCount, outChCount,
                new[] { kWidth, kHeight },
                new[] { strideX, strideY },
                new[] { padX, padY },
                false, w, b, gpuEnable: gpuEnable);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { inChCount, heightSize, wideSize }, batchCount);

            NdArray y = convolution2D.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,,,])cY.Data.Copy());
            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad.Copy());

            Real[] cWgrad = Real.ToRealArray((Real[,,,])cConvolotion2D.W.Grad);
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
