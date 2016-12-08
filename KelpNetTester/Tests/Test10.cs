using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet;
using KelpNet.Common;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Loss;
using KelpNet.Optimizers;
using VocabularyMaker;

namespace KelpNetTester.Tests
{
    //ChainerのRNNサンプルを再現
    //https://github.com/pfnet/chainer/tree/master/examples/ptb
    class Test10
    {
        const int N_EPOCH = 39;
        const int N_UNITS = 650;
        const int BATCH_SIZE = 20;
        const int BPROP_LEN = 35;
        const int GRAD_CLIP = 5;

        public static void Run()
        {
            Console.WriteLine("Build Vocabulary.");

            Vocabulary vocabulary = new Vocabulary();

            int[] trainData = vocabulary.LoadData("data/ptb.train.txt");
            int[] validData = vocabulary.LoadData("data/ptb.valid.txt");
            int[] testData = vocabulary.LoadData("data/ptb.test.txt");

            int nVocab = vocabulary.Length;

            Console.WriteLine("Network Initilizing.");
            FunctionStack model = new FunctionStack(
                new EmbedID(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l2 LSTM"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l3 LSTM"),
                new Dropout(),
                new Linear(N_UNITS, nVocab, name: "l4 Linear")
            );

            //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
            GradientClipping gradientClipping = new GradientClipping(threshold: GRAD_CLIP);
            SGD sgd = new SGD(learningRate: 1.0);
            model.SetOptimizer(gradientClipping, sgd);

            double wholeLen = trainData.Length;
            int jump = (int)Math.Floor(wholeLen / BATCH_SIZE);
            int epoch = 0;

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            Console.WriteLine("Train Start.");

            for (int i = 0; i < jump * N_EPOCH; i++)
            {

                int[][] x = new int[BATCH_SIZE][];
                int[][] t = new int[BATCH_SIZE][];

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x[j] = new[] { trainData[(int)((jump * j + i) % wholeLen)] };
                    t[j] = new[] { trainData[(int)((jump * j + i + 1) % wholeLen)] };
                }

                double sumLoss;
                backNdArrays.Push(Trainer.Forward(model, x, t, LossFunctions.SoftmaxCrossEntropy, out sumLoss));
                Console.WriteLine("[{0}/{1}] Loss: {2}", i + 1, jump, sumLoss);

                //Run truncated BPTT
                if ((i + 1) % BPROP_LEN == 0)
                {
                    for (int j = 0; backNdArrays.Count > 0; j++)
                    {
                        Console.WriteLine("backward" + backNdArrays.Count);
                        model.Backward(backNdArrays.Pop());
                    }

                    model.Update();
                    model.ResetState();
                }

                if ((i + 1) % jump == 0)
                {
                    epoch++;
                    Console.WriteLine("evaluate");
                    Console.WriteLine("validation perplexity: {0}", Evaluate(model, validData));

                    if (epoch >= 6)
                    {
                        sgd.LearningRate /= 1.2;
                        Console.WriteLine("learning rate =" + sgd.LearningRate);
                    }
                }
            }

            Console.WriteLine("test start");
            double testPerp = Evaluate(model, testData);
            Console.WriteLine("test perplexity:" + testPerp);
        }

        static double Evaluate(FunctionStack model, int[] dataset)
        {
            FunctionStack predictModel = model.Clone();
            predictModel.ResetState();

            List<double> totalLoss = new List<double>();

            for (int i = 0; i < dataset.Length - 1; i++)
            {
                int[][] x = new int[BATCH_SIZE][];
                int[][] t = new int[BATCH_SIZE][];

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x[j] = new[] { dataset[j + i] };
                    t[j] = new[] { dataset[j + i + 1] };
                }

                double sumLoss;
                Trainer.Forward(predictModel, x, t, LossFunctions.SoftmaxCrossEntropy, out sumLoss);
                totalLoss.Add(sumLoss);
            }

            //calc perplexity
            return Math.Exp(totalLoss.Sum() / (totalLoss.Count - 1));
        }
    }
}
