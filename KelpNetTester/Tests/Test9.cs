using System;
using System.Collections.Generic;
using KelpNet;
using KelpNet.Common;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;
using VocabularyMaker;

namespace KelpNetTester.Tests
{
    //RNNの実装
    //『Chainerによる実践深層学習』より（ISBN 978-4-274-21934-4）
    class Test9
    {
        const int TRAINING_EPOCHS = 5;
        const int N_UNITS = 100;

        public static void Run()
        {

            Console.WriteLine("Build Vocabulary.");

            Vocabulary vocabulary = new Vocabulary();
            var trainData = vocabulary.LoadData("data/ptb.train.txt");
            var testData = vocabulary.LoadData("data/ptb.test.txt");

            int nVocab = vocabulary.Length;

            Console.WriteLine("Done.");

            Console.WriteLine("Network Initilizing.");
            FunctionStack model = new FunctionStack(
                new EmbedID(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Linear(N_UNITS, N_UNITS, name: "l2 Linear"),
                new Tanh("l2 Tanh"),
                new Linear(N_UNITS, nVocab, name: "l3 Linear"),
                new Softmax("l3 Sonftmax")
            );

            model.SetOptimizer(new Adam());

            List<int> s = new List<int>();

            Console.WriteLine("Train Start.");
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                NdArray h = NdArray.Empty(N_UNITS);
                for (int pos = 0; pos < trainData.Length; pos++)
                {
                    var id = trainData[pos];
                    s.Add(id);

                    if (id == vocabulary.EosID)
                    {
                        double accumloss = 0;
                        Stack<NdArray> tmp = new Stack<NdArray>();

                        for (int i = 0; i < s.Count; i++)
                        {
                            var tx = i == s.Count - 1 ? vocabulary.EosID : s[i + 1];
                            //l1 Linear
                            var xK = model.Functions[0].Forward(NdArray.FromArray(new[] { s[i] }));

                            //l2 Linear
                            var l2 = model.Functions[1].Forward(h);
                            for (int j = 0; j < xK.Length; j++)
                            {
                                xK.Data[j] += l2.Data[j];
                            }

                            //l2 Tanh
                            h = model.Functions[2].Forward(xK);

                            //l3 Linear
                            var h2 = model.Functions[3].Forward(h);

                            double loss;
                            tmp.Push(LossFunctions.SoftmaxCrossEntropy(h2, NdArray.FromArray(new[] { tx }), out loss));
                            accumloss += loss;
                        }

                        Console.WriteLine(accumloss);

                        for (int i = 0; i < s.Count; i++)
                        {
                            var g = model.Functions[3].Backward(tmp.Pop());
                            g = model.Functions[2].Backward(g);
                            g = model.Functions[1].Backward(g);
                            model.Functions[0].Backward(g);
                        }

                        model.Update();
                        s.Clear();
                    }

                    if (pos % 100 == 0)
                    {
                        Console.WriteLine(pos + "/" + trainData.Length + " finished");
                    }
                }
            }

            Console.WriteLine("Test Start.");

            double sum = 0.0;
            int wnum = 0;
            List<int> ts = new List<int>();
            bool unkWord = false;

            for (int pos = 0; pos < 1000; pos++)
            {
                var id = testData[pos];
                ts.Add(id);

                if (id > trainData.Length)
                {
                    unkWord = true;
                }

                if (id == vocabulary.EosID)
                {
                    if (!unkWord)
                    {
                        Console.WriteLine("pos" + pos);
                        Console.WriteLine("tsLen" + ts.Count);
                        Console.WriteLine("sum" + sum);
                        Console.WriteLine("wnum" + wnum);

                        sum += CalPs(model, ts);
                        wnum += ts.Count - 1;
                    }
                    else
                    {
                        unkWord = false;
                    }

                    ts.Clear();
                }
            }

            Console.WriteLine(Math.Pow(2.0, sum / wnum));
        }

        static double CalPs(FunctionStack model, List<int> s)
        {
            double sum = 0.0;

            NdArray h = NdArray.Empty(N_UNITS);

            for (int i = 1; i < s.Count; i++)
            {
                //l1 Linear
                var xK = model.Functions[0].Forward(NdArray.FromArray(new[] { s[i] }));

                //l2 Linear
                var l2 = model.Functions[1].Forward(h);
                for (int j = 0; j < xK.Length; j++)
                {
                    xK.Data[j] += l2.Data[j];
                }

                //l2 Tanh
                h = model.Functions[2].Forward(xK);

                //l3 Softmax(l3 Linear)
                var yv = model.Functions[4].Forward(model.Functions[3].Forward(h));
                var pi = yv.Data[s[i - 1]];
                sum -= Math.Log(pi, 2);
            }

            return sum;
        }
    }
}

