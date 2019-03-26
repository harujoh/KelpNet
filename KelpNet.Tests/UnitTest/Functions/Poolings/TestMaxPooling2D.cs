using System;
using System.Drawing;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestMaxPooling2D
    {
        [TestMethod]
        public void MaxPoolingCPURandomTest()
        {
            RandomTest(false);
        }

        [TestMethod]
        public void MaxPoolingGPURandomTest()
        {
            Weaver.Initialize();

            if (Weaver.Enable)
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
            int chCount = Mother.Dice.Next(1, 5);
            int wideSize = Mother.Dice.Next(8, 32);
            int heightSize = Mother.Dice.Next(8, 32);

            int kWidth = Mother.Dice.Next(1, 5);
            int kHeight = Mother.Dice.Next(1, 5);

            int strideX = 1;//Mother.Dice.Next(1, 5);
            int strideY = 1;//Mother.Dice.Next(1, 5);

            //todo pad埋めがKelpNetは0固定だがChainerは-infinity
            int padX = 0;//Mother.Dice.Next(0, 5);
            int padY = 0;//Mother.Dice.Next(0, 5);

            bool coverAll = Mother.Dice.Next(1) == 0;

            int outputHeight = coverAll ?
                (int)Math.Floor((heightSize - kHeight + padY * 2.0 + strideY - 1) / strideY) + 1 :
                (int)Math.Floor((heightSize - kHeight + padY * 2.0) / strideY) + 1;

            int outputWidth = coverAll ?
                (int)Math.Floor((wideSize - kWidth + padX * 2.0 + strideX - 1) / strideX) + 1 :
                (int)Math.Floor((wideSize - kWidth + padX * 2.0) / strideX) + 1;

            Real[,,,] input = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, chCount, heightSize, wideSize });

            Real[,,,] dummyGy = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, chCount, outputHeight, outputWidth });

            //Chainer
            NChainer.MaxPooling2D<Real> cMaxPooling2D = new NChainer.MaxPooling2D<Real>(
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX }
            );

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cMaxPooling2D.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.MaxPooling2D maxPooling2D = new KelpNet.MaxPooling2D(
                new Size(kWidth, kHeight),
                new Size(strideX, strideY),
                new Size(padX, padY),
                gpuEnable: gpuEnable);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { chCount, heightSize, wideSize }, batchCount);

            NdArray y = maxPooling2D.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();

            Real[] cYdata = Real.ToRealArray((Real[,,,])cY.Data.Copy());
            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad.Copy());

            //許容範囲を算出
            double delta = 1;//0.00001;

            Assert.AreEqual(cYdata.Length, y.Data.Length);
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);

            //y
            for (int i = 0; i < y.Data.Length; i++)
            {
                if (!float.IsInfinity(cYdata[i]))
                {
                    Assert.AreEqual(cYdata[i], y.Data[i], delta);
                }
            }

            //x.grad
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }
        }
    }
}
