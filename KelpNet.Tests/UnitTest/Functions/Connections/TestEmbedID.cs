using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestEmbedID
    {
        [TestMethod]
        public void EmbedIDRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int inputCount = Mother.Dice.Next(2, 30);
            int outputCount = Mother.Dice.Next(1, 30);
            int batchCount = Mother.Dice.Next(1, 5);

            int[,] input = (int[,])Enumerable.Repeat(0, batchCount * inputCount).ToNdArray(batchCount, inputCount);
            input[0, 0] = 1;

            Real[,,] dummyGy = (Real[,,])Initializer.GetRealNdArray(new[] { batchCount, inputCount, outputCount });
            Real[,] w = (Real[,])Initializer.GetRealNdArray(new[] { inputCount, outputCount });

            //Chainer
            NChainer.EmbedID<Real> cEmbedId = new NChainer.EmbedID<Real>(inputCount, outputCount, Real.ToBaseNdArray(w));

            Variable<int> cX = new Variable<int>(input);

            Variable<Real> cY = cEmbedId.Forward(cX);
            cY.Grad = Real.ToBaseNdArray(dummyGy);

            cY.Backward();


            //KelpNet
            KelpNet.EmbedID embedId = new KelpNet.EmbedID(inputCount, outputCount, w);

            NdArray x = new NdArray(Real.ToRealArray(input), new[] { inputCount }, batchCount);

            NdArray y = embedId.Forward(x)[0];
            y.Grad = Real.ToRealArray(dummyGy);

            y.Backward();


            Real[] cYdata = Real.ToRealArray((Real[,,])cY.Data);

            Real[] cWgrad = Real.ToRealArray((Real[,])cEmbedId.W.Grad);

            //許容範囲を算出
            double delta = 0.00001;

            //y
            Assert.AreEqual(cYdata.Length, y.Data.Length);
            for (int i = 0; i < y.Data.Length; i++)
            {
                Assert.AreEqual(cYdata[i], y.Data[i], delta);
            }

            //W.grad
            Assert.AreEqual(cWgrad.Length, embedId.Weight.Grad.Length);
            for (int i = 0; i < embedId.Weight.Grad.Length; i++)
            {
                Assert.AreEqual(cWgrad[i], embedId.Weight.Grad[i], delta);
            }
        }
    }
}
