using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

#if DOUBLE
#elif NETCOREAPP2_0
using Math = System.MathF;
#else
using Math = KelpNet.MathF;
#endif

//using Real = System.Double;
using Real = System.Single;

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

            int strideX = Mother.Dice.Next(1, 5);
            int strideY = Mother.Dice.Next(1, 5);

            int padX = Mother.Dice.Next(0, 5);
            int padY = Mother.Dice.Next(0, 5);


            int outputHeight = (int)Math.Floor((heightSize - kHeight + padY * 2.0f) / strideY) + 1;

            int outputWidth = (int)Math.Floor((wideSize - kWidth + padX * 2.0f) / strideX) + 1;

            Real[,,,] input = Initializer.GetRandomValues<Real[,,,]>(batchCount, chCount, heightSize, wideSize);

            Real[,,,] dummyGy = Initializer.GetRandomValues<Real[,,,]>(batchCount, chCount, outputHeight, outputWidth);

            //Chainer
            NChainer.AveragePooling2D<Real> cMaxPooling2D = new NChainer.AveragePooling2D<Real>(
                new[] { kHeight, kWidth },
                new[] { strideY, strideX },
                new[] { padY, padX }
            );

            Variable<Real> cX = new Variable<Real>(input);

            Variable<Real> cY = cMaxPooling2D.Forward(cX);
            cY.Grad = dummyGy;

            cY.Backward();


            //KelpNet
            KelpNet.AveragePooling2D<Real> maxPooling2D = new KelpNet.AveragePooling2D<Real>(
                new[] { kWidth, kHeight },
                new[] { strideX, strideY },
                new[] { padX, padY });

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);

            NdArray<Real> y = maxPooling2D.Forward(x)[0];
            y.Grad = dummyGy.Flatten();

            y.Backward();

            Real[] cYdata = ((Real[,,,])cY.Data.Copy()).Flatten();
            Real[] cXgrad = ((Real[,,,])cX.Grad.Copy()).Flatten();

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
