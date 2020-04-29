using KelpNet.CL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestDeconvolution2D
    {
        [TestMethod]
        public void Deconv2DCPURandomTest()
        {
            RandomTest(false);
        }

        [TestMethod]
        public void Deconv2DGPURandomTest()
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

            int outputHeight = (heightSize - 1) * strideY + kHeight - padY * 2;
            int outputWidth = (wideSize - 1) * strideX + kWidth - padX * 2;

            Real[,,,] input = Initializer.GetRandomValues<Real[,,,]>(batchCount, inChCount, heightSize, wideSize);

            Real[,,,] dummyGy = Initializer.GetRandomValues<Real[,,,]>(batchCount, outChCount, outputHeight, outputWidth);
            Real[,,,] w = Initializer.GetRandomValues<Real[,,,]>(inChCount, outChCount, kHeight, kWidth);

            Real[] b = Initializer.GetRandomValues<Real[]>(outChCount);

            //Chainer
            NChainer.Deconvolution2D<Real> cDeconvolotion2D = new NChainer.Deconvolution2D<Real>(
                inChCount, outChCount,
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX },
                false,
                new PyObject[] { outputHeight, outputWidth },
                w,
                b);

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cDeconvolotion2D.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();


            //KelpNet
            CL.Deconvolution2D<Real> deconvolution2D = new CL.Deconvolution2D<Real>(
                inChCount, outChCount,
                new []{kWidth, kHeight},
                new []{strideX, strideY},
                new []{padX, padY},
                false, w, b, gpuEnable: gpuEnable);

            NdArray<Real> x = new NdArray<Real>(input, asBatch:true);

            NdArray<Real> y = deconvolution2D.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();


            Real[] cYdata = ((Real[,,,])cY.Data.Copy()).Flatten();
            Real[] cXgrad = ((Real[,,,])cX.Grad.Copy()).Flatten();

            Real[] cWgrad = ((Real[,,,])cDeconvolotion2D.W.Grad).Flatten();
            Real[] cbgrad = (Real[])cDeconvolotion2D.b.Grad;

            //許容範囲を算出
            double delta = 0.00001;

            Assert.AreEqual(cYdata.Length, y.Data.Length);
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            Assert.AreEqual(cWgrad.Length, deconvolution2D.Weight.Grad.Length);
            Assert.AreEqual(cbgrad.Length, deconvolution2D.Bias.Grad.Length);

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
            for (int i = 0; i < deconvolution2D.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad[i], deconvolution2D.Weight.Grad[i], delta);
            }

            //b.grad
            for (int i = 0; i < deconvolution2D.Bias.Grad.Length; i++)
            {
                Assert.AreEqual(cbgrad[i], deconvolution2D.Bias.Grad[i], delta);
            }
        }
    }
}
