using System;
using KelpNet.Common;
using KelpNet.Functions.Connections;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //Linearの分割テスト
    class TestX
    {
        public static void Run()
        {
            Linear l0 = new Linear(5, 20, initialW: new[,]{
                {(Real)(-0.02690255),(Real)0.08830735,(Real)(-0.02041466),(Real)(-0.0431439),(Real)(-0.07749002)},
                {(Real)(-0.06963444),(Real)(-0.03971611),(Real)0.0597842,(Real)0.08824182,(Real)(-0.06649109)},
                {(Real)(-0.04966073),(Real)(-0.04697048),(Real)(-0.02235234),(Real)(-0.09396666),(Real)0.073189},
                {(Real)0.06563969,(Real)0.04446745,(Real)(-0.07192299),(Real)0.06784364,(Real)0.09575776},
                {(Real)0.05012317,(Real)(-0.08874852),(Real)(-0.05977172),(Real)(-0.05910181),(Real)(-0.06009106)},
                {(Real)(-0.05200623),(Real)(-0.09679124),(Real)0.02159978,(Real)(-0.08058041),(Real)(-0.01340541)},
                {(Real)(-0.0254951),(Real)0.09963084,(Real)0.00936683,(Real)(-0.08179696),(Real)0.09604459},
                {(Real)(-0.0732494),(Real)0.07253634,(Real)0.05981455,(Real)(-0.01007657),(Real)(-0.02992892)},
                {(Real)(-0.06818873),(Real)(-0.02579817),(Real)0.06767359,(Real)(-0.03379837),(Real)(-0.04880046)},
                {(Real)(-0.06429326),(Real)(-0.08964688),(Real)(-0.0960066),(Real)(-0.00286683),(Real)(-0.05761427)},
                {(Real)(-0.0454098),(Real)0.07809167,(Real)(-0.05030088),(Real)(-0.02533244),(Real)(-0.02322736)},
                {(Real)(-0.00866754),(Real)(-0.03614252),(Real)0.05237325,(Real)0.06478979,(Real)(-0.03599609)},
                {(Real)(-0.01789357),(Real)(-0.04479434),(Real)(-0.05765592),(Real)0.03237658,(Real)(-0.06403019)},
                {(Real)(-0.02421552),(Real)0.05533903,(Real)(-0.08627617),(Real)0.094624,(Real)0.03319318},
                {(Real)0.02328842,(Real)(-0.08234859),(Real)(-0.07979888),(Real)0.01439688,(Real)(-0.03267198)},
                {(Real)(-0.07128382),(Real)0.08531934,(Real)0.07180037,(Real)0.04772871,(Real)(-0.08938966)},
                {(Real)0.09431138,(Real)0.02094762,(Real)0.04443646,(Real)0.07653841,(Real)0.02028433},
                {(Real)0.01844446,(Real)(-0.08441339),(Real)0.01957355,(Real)0.04430714,(Real)(-0.03080243)},
                {(Real)(-0.0261334),(Real)(-0.03794889),(Real)(-0.00638074),(Real)0.07278767,(Real)(-0.02165155)},
                {(Real)0.08390063,(Real)(-0.03253863),(Real)0.0311571,(Real)0.08088892,(Real)(-0.07267931)}
            }, name: "l0");

            Linear l1 = new Linear(5, 5, initialW: new[,]{
                {(Real)(-0.02690255),(Real)0.08830735,(Real)(-0.02041466),(Real)(-0.0431439),(Real)(-0.07749002)},
                {(Real)(-0.06963444),(Real)(-0.03971611),(Real)0.0597842,(Real)0.08824182,(Real)(-0.06649109)},
                {(Real)(-0.04966073),(Real)(-0.04697048),(Real)(-0.02235234),(Real)(-0.09396666),(Real)0.073189},
                {(Real)0.06563969,(Real)0.04446745,(Real)(-0.07192299),(Real)0.06784364,(Real)0.09575776},
                {(Real)0.05012317,(Real)(-0.08874852),(Real)(-0.05977172),(Real)(-0.05910181),(Real)(-0.06009106)}
            }, name: "l1");

            Linear l2 = new Linear(5, 5, initialW: new[,]{
                {(Real)(-0.05200623),(Real)(-0.09679124),(Real)0.02159978,(Real)(-0.08058041),(Real)(-0.01340541)},
                {(Real)(-0.0254951),(Real)0.09963084,(Real)0.00936683,(Real)(-0.08179696),(Real)0.09604459},
                {(Real)(-0.0732494),(Real)0.07253634,(Real)0.05981455,(Real)(-0.01007657),(Real)(-0.02992892)},
                {(Real)(-0.06818873),(Real)(-0.02579817),(Real)0.06767359,(Real)(-0.03379837),(Real)(-0.04880046)},
                {(Real)(-0.06429326),(Real)(-0.08964688),(Real)(-0.0960066),(Real)(-0.00286683),(Real)(-0.05761427)}
            }, name: "l2");

            Linear l3 = new Linear(5, 5, initialW: new[,]{
                {(Real)(-0.0454098),(Real)0.07809167,(Real)(-0.05030088),(Real)(-0.02533244),(Real)(-0.02322736)},
                {(Real)(-0.00866754),(Real)(-0.03614252),(Real)0.05237325,(Real)0.06478979,(Real)(-0.03599609)},
                {(Real)(-0.01789357),(Real)(-0.04479434),(Real)(-0.05765592),(Real)0.03237658,(Real)(-0.06403019)},
                {(Real)(-0.02421552),(Real)0.05533903,(Real)(-0.08627617),(Real)0.094624,(Real)0.03319318},
                {(Real)0.02328842,(Real)(-0.08234859),(Real)(-0.07979888),(Real)0.01439688,(Real)(-0.03267198)}
            }, name: "l3");

            Linear l4 = new Linear(5, 5, initialW: new[,]{
                {(Real)(-0.07128382),(Real)0.08531934,(Real)0.07180037,(Real)0.04772871,(Real)(-0.08938966)},
                {(Real)0.09431138,(Real)0.02094762,(Real)0.04443646,(Real)0.07653841,(Real)0.02028433},
                {(Real)0.01844446,(Real)(-0.08441339),(Real)0.01957355,(Real)0.04430714,(Real)(-0.03080243)},
                {(Real)(-0.0261334),(Real)(-0.03794889),(Real)(-0.00638074),(Real)0.07278767,(Real)(-0.02165155)},
                {(Real)0.08390063,(Real)(-0.03253863),(Real)0.0311571,(Real)0.08088892,(Real)(-0.07267931)}
            }, name: "l4");


            //FunctionにOptimiserを設定
            SGD sgd = new SGD();
            l0.SetOptimizer(sgd);
            l1.SetOptimizer(sgd);
            l2.SetOptimizer(sgd);
            l3.SetOptimizer(sgd);
            l4.SetOptimizer(sgd);


            Console.WriteLine("l0 for");
            Console.WriteLine(l0.Forward(new BatchArray(new[] { (Real)0.01618112, (Real)(-0.08296648), (Real)(-0.05545357), (Real)0.00389254, (Real)(-0.05727582) })));
            Console.WriteLine("\nl1 for");
            Console.WriteLine(l1.Forward(new BatchArray(new[] { (Real)0.01618112, (Real)(-0.08296648), (Real)(-0.05545357), (Real)0.00389254, (Real)(-0.05727582) })));
            Console.WriteLine("\nl2 for");
            Console.WriteLine(l2.Forward(new BatchArray(new[] { (Real)0.01618112, (Real)(-0.08296648), (Real)(-0.05545357), (Real)0.00389254, (Real)(-0.05727582) })));
            Console.WriteLine("\nl3 for");
            Console.WriteLine(l3.Forward(new BatchArray(new[] { (Real)0.01618112, (Real)(-0.08296648), (Real)(-0.05545357), (Real)0.00389254, (Real)(-0.05727582) })));
            Console.WriteLine("\nl4 for");
            Console.WriteLine(l4.Forward(new BatchArray(new[] { (Real)0.01618112, (Real)(-0.08296648), (Real)(-0.05545357), (Real)0.00389254, (Real)(-0.05727582) })));
            Console.WriteLine();

            Console.WriteLine("\nl0 back");
            Console.WriteLine(l0.Backward(new BatchArray(new[,]
            {
                {(Real)(-2.42022760e-02),(Real) 5.02482988e-04,(Real) 2.52015481e-04,(Real) 8.08797951e-04},
                {(Real)(-7.19293347e-03),(Real) 1.40045900e-04,(Real) 7.09874439e-05,(Real) 2.07651625e-04},
                {(Real)3.80124636e-02,(Real)(- 8.87162634e-04),(Real)(- 4.64874669e-04),(Real)(- 1.40792923e-03)},
                {(Real)(-4.12280299e-02),(Real)(- 3.36557830e-04),(Real)(- 1.50323089e-04),(Real)(- 4.70047118e-04)},
                {(Real)3.61101292e-02,(Real)(- 7.12957408e-04),(Real)(- 3.63163825e-04),(Real)(- 1.12809543e-03)}
            })));

            BatchArray l1bak = l1.Backward(new BatchArray(new[]
            {
                (Real)(-2.42022760e-02),(Real)5.02482988e-04,(Real) 2.52015481e-04,(Real)8.08797951e-04,(Real)(- 7.19293347e-03)
            }));

            BatchArray l2bak = l2.Backward(new BatchArray(new[]
            {
                (Real)1.40045900e-04,(Real)7.09874439e-05,(Real) 2.07651625e-04,(Real)3.80124636e-02,(Real)(- 8.87162634e-04)
            }));

            BatchArray l3bak = l3.Backward(new BatchArray(new[]
            {
                (Real)(-4.64874669e-04),(Real)(- 1.40792923e-03),(Real)(- 4.12280299e-02),(Real)(- 3.36557830e-04),(Real)(- 1.50323089e-04)
            }));

            BatchArray l4bak = l4.Backward(new BatchArray(new[]
            {
                (Real)(-4.70047118e-04), (Real)3.61101292e-02,(Real)(-7.12957408e-04), (Real)(-3.63163825e-04), (Real)(-1.12809543e-03)
            }));

            Console.WriteLine("\nl1-l4 sum back");

            NdArray lsum = new NdArray(l1bak.Shape);
            for (int i = 0; i < lsum.Data.Length; i++)
            {
                lsum.Data[i] += l1bak.Data[i] + l2bak.Data[i] + l3bak.Data[i] + l4bak.Data[i];
            }
            Console.WriteLine(lsum);

            sgd.Update();

            Console.WriteLine("\nl0 W");
            Console.WriteLine(l0.W);

            Console.WriteLine("\nl1 W");
            Console.WriteLine(l1.W);

            Console.WriteLine("\nl0 b");
            Console.WriteLine(l0.b);

            Console.WriteLine("\nl1 b");
            Console.WriteLine(l1.b);
        }
    }
}
