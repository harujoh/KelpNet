using System;
using System.Drawing;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestAveragePooling2D
    {
        [TestMethod]
        public void AVGPoolingRandomTest()
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

            int padX = 0;//Mother.Dice.Next(0, 5);
            int padY = 0;//Mother.Dice.Next(0, 5);


            int outputHeight = (int)Math.Floor((heightSize - kHeight + padY * 2.0) / strideY) + 1;

            int outputWidth = (int)Math.Floor((wideSize - kWidth + padX * 2.0) / strideX) + 1;

            Real[,,,] input = (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, chCount, heightSize, wideSize });

            Real[,,,] dummyGy =
                (Real[,,,])Initializer.GetRealNdArray(new[] { batchCount, chCount, outputHeight, outputWidth });

            //Chainer
            NChainer.AveragePooling2D<Real> cMaxPooling2D = new NChainer.AveragePooling2D<Real>(
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX }
            );

            Variable<Real> cX = new Variable<Real>(Real.ToBaseNdArray(input));

            Variable<Real> cY = cMaxPooling2D.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.AveragePooling2D maxPooling2D = new KelpNet.AveragePooling2D(
                new Size(kWidth, kHeight),
                new Size(strideX, strideY),
                new Size(padX, padY));

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { chCount, heightSize, wideSize }, batchCount);

            NdArray y = maxPooling2D.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();

            Real[] cYdata = Real.ToRealArray((Real[,,,])cY.Data.Copy());
            Real[] cXgrad = Real.ToRealArray((Real[,,,])cX.Grad.Copy());

            //許容範囲を算出
            double delta = 0.00001;

            Assert.AreEqual(cYdata.Length, y.Data.Length);
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);

            //y
            for (int i = 0; i < y.Data.Length; i++)
            {
                if (cYdata[i] < float.Epsilon && cYdata[i] > -float.Epsilon)
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
