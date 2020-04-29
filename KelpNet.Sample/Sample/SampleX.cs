using System;
using KelpNet.CL;

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    //Linearの分割テスト
    class SampleX
    {
        public static void Run()
        {
            //Weightを分割の前と後で揃える
            Real[,] testWeightValues = new Real[,]{
                            {-0.02690255f, 0.08830735f, -0.02041466f, -0.0431439f, -0.07749002f},
                            {-0.06963444f, -0.03971611f, 0.0597842f, 0.08824182f, -0.06649109f},
                            {-0.04966073f, -0.04697048f, -0.02235234f, -0.09396666f, 0.073189f},
                            {0.06563969f, 0.04446745f, -0.07192299f, 0.06784364f, 0.09575776f},
                            {0.05012317f, -0.08874852f, -0.05977172f, -0.05910181f, -0.06009106f},
                            {-0.05200623f, -0.09679124f, 0.02159978f, -0.08058041f, -0.01340541f},
                            {-0.0254951f, 0.09963084f, 0.00936683f, -0.08179696f, 0.09604459f},
                            {-0.0732494f, 0.07253634f, 0.05981455f, -0.01007657f, -0.02992892f},
                            {-0.06818873f, -0.02579817f, 0.06767359f, -0.03379837f, -0.04880046f},
                            {-0.06429326f, -0.08964688f, -0.0960066f, -0.00286683f, -0.05761427f},
                            {-0.0454098f, 0.07809167f, -0.05030088f, -0.02533244f, -0.02322736f},
                            {-0.00866754f, -0.03614252f, 0.05237325f, 0.06478979f, -0.03599609f},
                            {-0.01789357f, -0.04479434f, -0.05765592f, 0.03237658f, -0.06403019f},
                            {-0.02421552f, 0.05533903f, -0.08627617f, 0.094624f, 0.03319318f},
                            {0.02328842f, -0.08234859f, -0.07979888f, 0.01439688f, -0.03267198f},
                            {-0.07128382f, 0.08531934f, 0.07180037f, 0.04772871f, -0.08938966f},
                            {0.09431138f, 0.02094762f, 0.04443646f, 0.07653841f, 0.02028433f},
                            {0.01844446f, -0.08441339f, 0.01957355f, 0.04430714f, -0.03080243f},
                            {-0.0261334f, -0.03794889f, -0.00638074f, 0.07278767f, -0.02165155f},
                            {0.08390063f, -0.03253863f, 0.0311571f, 0.08088892f, -0.07267931f}
            };

            Real[][,] testJaggWeightValues = {
                new Real[,] {{-0.02690255f, 0.08830735f, -0.02041466f, -0.0431439f, -0.07749002f},
                             {-0.06963444f, -0.03971611f, 0.0597842f, 0.08824182f, -0.06649109f},
                             {-0.04966073f, -0.04697048f, -0.02235234f, -0.09396666f, 0.073189f},
                             {0.06563969f, 0.04446745f, -0.07192299f, 0.06784364f, 0.09575776f},
                             {0.05012317f, -0.08874852f, -0.05977172f, -0.05910181f, -0.06009106f}},
                new Real[,] {{-0.05200623f, -0.09679124f, 0.02159978f, -0.08058041f, -0.01340541f},
                             {-0.0254951f, 0.09963084f, 0.00936683f, -0.08179696f, 0.09604459f},
                             {-0.0732494f, 0.07253634f, 0.05981455f, -0.01007657f, -0.02992892f},
                             {-0.06818873f, -0.02579817f, 0.06767359f, -0.03379837f, -0.04880046f},
                             {-0.06429326f, -0.08964688f, -0.0960066f, -0.00286683f, -0.05761427f}},
                new Real[,] {{-0.0454098f, 0.07809167f, -0.05030088f, -0.02533244f, -0.02322736f},
                             {-0.00866754f, -0.03614252f, 0.05237325f, 0.06478979f, -0.03599609f},
                             {-0.01789357f, -0.04479434f, -0.05765592f, 0.03237658f, -0.06403019f},
                             {-0.02421552f, 0.05533903f, -0.08627617f, 0.094624f, 0.03319318f},
                             {0.02328842f, -0.08234859f, -0.07979888f, 0.01439688f, -0.03267198f}},
                new Real[,] {{-0.07128382f, 0.08531934f, 0.07180037f, 0.04772871f, -0.08938966f},
                             {0.09431138f, 0.02094762f, 0.04443646f, 0.07653841f, 0.02028433f},
                             {0.01844446f, -0.08441339f, 0.01957355f, 0.04430714f, -0.03080243f},
                             {-0.0261334f, -0.03794889f, -0.00638074f, 0.07278767f, -0.02165155f},
                             {0.08390063f, -0.03253863f, 0.0311571f, 0.08088892f, -0.07267931f}}
            };

            Linear<Real> l0 = new Linear<Real>(5, 20, initialW: testWeightValues, name: "l0");

            Linear<Real> l1 = new Linear<Real>(5, 5, initialW: testJaggWeightValues[0], name: "l1");
            Linear<Real> l2 = new Linear<Real>(5, 5, initialW: testJaggWeightValues[1], name: "l2");
            Linear<Real> l3 = new Linear<Real>(5, 5, initialW: testJaggWeightValues[2], name: "l3");
            Linear<Real> l4 = new Linear<Real>(5, 5, initialW: testJaggWeightValues[3], name: "l4");

            //FunctionにOptimizerを設定
            SGD<Real> sgd = new SGD<Real>();
            sgd.SetUp(l0);

            //OptimiserにFunctionを登録
            SGD<Real> sgdSplit = new SGD<Real>();
            sgdSplit.SetUp(l1);
            sgdSplit.SetUp(l2);
            sgdSplit.SetUp(l3);
            sgdSplit.SetUp(l4);


            //入力は同値だがGradが加算されてしまうため分ける
            Real[] testValue = new Real[] { 0.01618112f, -0.08296648f, -0.05545357f, 0.00389254f, -0.05727582f };
            NdArray<Real> testInputValuesA = new NdArray<Real>(testValue);
            NdArray<Real> testInputValuesB = new NdArray<Real>(testValue);

            Console.WriteLine("l0 for");
            NdArray<Real> l0Result = l0.Forward(testInputValuesA)[0];
            Console.WriteLine(l0Result);

            Console.WriteLine("\nl1 for");
            NdArray<Real> l1Result = l1.Forward(testInputValuesB)[0];
            Console.WriteLine(l1Result);

            Console.WriteLine("\nl2 for");
            NdArray<Real> l2Result = l2.Forward(testInputValuesB)[0];
            Console.WriteLine(l2Result);

            Console.WriteLine("\nl3 for");
            NdArray<Real> l3Result = l3.Forward(testInputValuesB)[0];
            Console.WriteLine(l3Result);

            Console.WriteLine("\nl4 for");
            NdArray<Real> l4Result = l4.Forward(testInputValuesB)[0];
            Console.WriteLine(l4Result);

            Console.WriteLine();

            //適当なGrad値をでっち上げる
            l0Result.Grad = new Real[]
                                    {
                                        -2.42022760e-02f, 5.02482988e-04f, 2.52015481e-04f, 8.08797951e-04f, -7.19293347e-03f,
                                        1.40045900e-04f, 7.09874439e-05f, 2.07651625e-04f, 3.80124636e-02f, -8.87162634e-04f,
                                        -4.64874669e-04f, -1.40792923e-03f, -4.12280299e-02f, -3.36557830e-04f, -1.50323089e-04f,
                                        -4.70047118e-04f, 3.61101292e-02f, -7.12957408e-04f, -3.63163825e-04f, -1.12809543e-03f
                                    };

            l1Result.Grad = new Real[] { -2.42022760e-02f, 5.02482988e-04f, 2.52015481e-04f, 8.08797951e-04f, -7.19293347e-03f };
            l2Result.Grad = new Real[] { 1.40045900e-04f, 7.09874439e-05f, 2.07651625e-04f, 3.80124636e-02f, -8.87162634e-04f };
            l3Result.Grad = new Real[] { -4.64874669e-04f, -1.40792923e-03f, -4.12280299e-02f, -3.36557830e-04f, -1.50323089e-04f };
            l4Result.Grad = new Real[] { -4.70047118e-04f, 3.61101292e-02f, -7.12957408e-04f, -3.63163825e-04f, -1.12809543e-03f };


            //Backwardを実行
            l0.Backward(l0Result);

            l1.Backward(l1Result);
            l2.Backward(l2Result);
            l3.Backward(l3Result);
            l4.Backward(l4Result);

            Console.WriteLine("\nl0 back");
            Console.WriteLine(testInputValuesA.ToString("Grad"));

            Console.WriteLine("\nl1-l4 sum back");
            Console.WriteLine(testInputValuesB.ToString("Grad"));

            sgd.Update();
            sgdSplit.Update();

            Console.WriteLine("\nl0 Weight");
            Console.WriteLine(l0.Weight);

            Console.WriteLine("\nl1 Weight");
            Console.WriteLine(l1.Weight);

            Console.WriteLine("\nl0 Bias");
            Console.WriteLine(l0.Bias);

            Console.WriteLine("\nl1 Bias");
            Console.WriteLine(l1.Bias);
        }
    }
}
