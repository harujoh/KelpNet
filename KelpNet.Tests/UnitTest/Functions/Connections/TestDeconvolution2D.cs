using KelpNet.CL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

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

            Real[,,,] input = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, inChCount, heightSize, wideSize });

            Real[,,,] dummyGy = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, outChCount, outputHeight, outputWidth });
            Real[,,,] w = (Real[,,,])Initializer.GetRealNdArray(new[] { inChCount, outChCount, kHeight, kWidth });

            Real[] b = Initializer.GetRealArray(outChCount);

            //Chainer
            NChainer.Deconvolution2D<Real> cDeconvolotion2D = new NChainer.Deconvolution2D<Real>(
                inChCount, outChCount,
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX },
                false,
                new PyObject[] { outputHeight, outputWidth },
                Real.ToBaseNdArray(w),
                Real.ToBaseArray(b));

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cDeconvolotion2D.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.CL.Deconvolution2D deconvolution2D = new KelpNet.CL.Deconvolution2D(
                inChCount, outChCount,
                new []{kWidth, kHeight},
                new []{strideX, strideY},
                new []{padX, padY},
                false, w, b, gpuEnable: gpuEnable);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { inChCount, heightSize, wideSize }, batchCount);

            NdArray y = deconvolution2D.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,,,])cY.Data.Copy());
            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad.Copy());

            Real[] cWgrad = Real.ToRealArray((Real[,,,])cDeconvolotion2D.W.Grad);
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
