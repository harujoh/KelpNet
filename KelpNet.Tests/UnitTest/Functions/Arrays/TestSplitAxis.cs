using Microsoft.VisualStudio.TestTools.UnitTesting;
using NChainer;
using NConstrictor;
//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Tests
{
    [TestClass]
    public class TestSplitAxis
    {
        [TestMethod]
        public void SplitAxisRandomTest()
        {
            Python.Initialize();
            Chainer.Initialize();

            int batchCount = Mother.Dice.Next(1, 5);
            int ch = Mother.Dice.Next(1, 5);
            int widthA = Mother.Dice.Next(1, 16);
            int widthB = Mother.Dice.Next(1, 16);
            int height = Mother.Dice.Next(1, 16);
            int axis = 3;

            Real[,,,] input = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, widthA + widthB);

            Real[,,,] dummyGyA = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, widthA);
            Real[,,,] dummyGyB = Initializer.GetRandomValues<Real[,,,]>(batchCount, ch, height, widthB);

            //chainer
            NChainer.SplitAxis<Real> cSplitAxis = new NChainer.SplitAxis<Real>();

            Variable<Real> cX = new Variable<Real>(input);
            PyObject[] cY = cSplitAxis.Forward(cX, new[] { widthA }, axis);

            Variable<Real> cY0 = cY[0];
            Variable<Real> cY1 = cY[1];

            cY0.Grad = dummyGyA;
            cY1.Grad = dummyGyB;

            //Chainerはどちらか一方で両方分のBackwardが走る
            cY0.Backward();
            //cY1.Backward();


            //KelpNet
            SplitAxis<Real> splitAxis = new SplitAxis<Real>(new[] { widthA }, axis - 1);//Chainerと異なり1次元目を暗黙的にBatchとみなさないため

            NdArray<Real> x = new NdArray<Real>(input, asBatch: true);// new NdArray(Real.ToRealArray(), new[] { ch, height, widthA + widthB }, batchCount);

            NdArray<Real>[] y = splitAxis.Forward(x);
            y[0].Grad = dummyGyA.Flatten();
            y[1].Grad = dummyGyB.Flatten();

            //KelpNetは出力した両方からBackwardしないと処理が走らない
            y[0].Backward();
            y[1].Backward();

            //Copyが必要
            Real[] cY0data = ((Real[,,,])cY0.Data.Copy()).Flatten();
            Real[] cY1data = ((Real[,,,])cY1.Data.Copy()).Flatten();

            Real[] cXgrad = ((Real[,,,])cX.Grad).Flatten();

            //許容範囲を算出
            double delta = 0.00001;

            //y0
            Assert.AreEqual(cY0data.Length, y[0].Data.Length);
            for (int i = 0; i < y[0].Data.Length; i++)
            {
                Assert.AreEqual(cY0data[i], y[0].Data[i], delta);
            }

            //y1
            Assert.AreEqual(cY1data.Length, y[1].Data.Length);
            for (int i = 0; i < y[1].Data.Length; i++)
            {
                Assert.AreEqual(cY1data[i], y[1].Data[i], delta);
            }

            //x.grad
            Assert.AreEqual(cXgrad.Length, x.Grad.Length);
            for (int i = 0; i < x.Grad.Length; i++)
            {
                Assert.AreEqual(cXgrad[i], x.Grad[i], delta);
            }
        }
    }
}
