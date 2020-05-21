using System;
using KelpNet.CL;
using KelpNet.Tools;

#if DOUBLE
#elif NETSTANDARD2_1
using Math = System.MathF;
#else
using Math = KelpNet.MathF;
#endif

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    //ChainerのRNNサンプルを再現
    //https://github.com/pfnet/chainer/tree/master/examples/ptb
    class Sample10
    {
        const int N_EPOCH = 39;
        const int N_UNITS = 650;
        const int BATCH_SIZE = 20;
        const int BPROP_LEN = 35;
        const int GRAD_CLIP = 5;

        const string DOWNLOAD_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/";

        const string TRAIN_FILE = "ptb.train.txt";
        const string VALID_FILE = "ptb.valid.txt";
        const string TEST_FILE = "ptb.test.txt";

        const string TRAIN_FILE_HASH = "f26c4b92c5fdc7b3f8c7cdcb991d8420";
        const string VALID_FILE_HASH = "aa0affc06ff7c36e977d7cd49e3839bf";
        const string TEST_FILE_HASH = "8b80168b89c18661a38ef683c0dc3721";

        public static void Run()
        {
            Console.WriteLine("Build Vocabulary.");

            Vocabulary vocabulary = new Vocabulary();

            string trainPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TRAIN_FILE, TRAIN_FILE, TRAIN_FILE_HASH);
            string validPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + VALID_FILE, VALID_FILE, VALID_FILE_HASH);
            string testPath = InternetFileDownloader.Donwload(DOWNLOAD_URL + TEST_FILE, TEST_FILE, TEST_FILE_HASH);

            int[] trainData = vocabulary.LoadData(trainPath);
            int[] validData = vocabulary.LoadData(validPath);
            int[] testData = vocabulary.LoadData(testPath);

            int nVocab = vocabulary.Length;

            Console.WriteLine("Network Initilizing.");
            FunctionStack<Real> model = new FunctionStack<Real>(
                new EmbedID<Real>(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Dropout<Real>(),
                new LSTM<Real>(N_UNITS, N_UNITS, name: "l2 LSTM"),
                new Dropout<Real>(),
                new LSTM<Real>(N_UNITS, N_UNITS, name: "l3 LSTM"),
                new Dropout<Real>(),
                new Linear<Real>(N_UNITS, nVocab, name: "l4 Linear")
            );

            for (int i = 0; i < model.Functions.Length; i++)
            {
                for (int j = 0; j < model.Functions[i].Parameters.Length; j++)
                {
                    for (int k = 0; k < model.Functions[i].Parameters[j].Data.Length; k++)
                    {
                        model.Functions[i].Parameters[j].Data[k] = ((Real)Mother.Dice.NextDouble() * 2.0f - 1.0f) / 10.0f;
                    }
                }
            }

            //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
            GradientClipping<Real> gradientClipping = new GradientClipping<Real>(threshold: GRAD_CLIP);
            SGD<Real> sgd = new SGD<Real>(learningRate: 0.1f);
            gradientClipping.SetUp(model);
            sgd.SetUp(model);

            Real wholeLen = trainData.Length;
            int jump = (int)Math.Floor(wholeLen / BATCH_SIZE);
            int epoch = 0;

            Console.WriteLine("Train Start.");

            for (int i = 0; i < jump * N_EPOCH; i++)
            {
                NdArray<Real> x = new NdArray<Real>(new[] { 1 }, BATCH_SIZE);
                NdArray<int> t = new NdArray<int>(new[] { 1 }, BATCH_SIZE);

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x.Data[j] = trainData[(int)((jump * j + i) % wholeLen)];
                    t.Data[j] = trainData[(int)((jump * j + i + 1) % wholeLen)];
                }

                NdArray<Real> result = model.Forward(x)[0];
                Real sumLoss = new SoftmaxCrossEntropy<Real>().Evaluate(result, t);
                Console.WriteLine("[{0}/{1}] Loss: {2}", i + 1, jump, sumLoss);
                model.Backward(result);

                //Run truncated BPTT
                if ((i + 1) % BPROP_LEN == 0)
                {
                    gradientClipping.Update();
                    sgd.Update();
                    model.ResetState();
                }

                if ((i + 1) % jump == 0)
                {
                    epoch++;
                    Console.WriteLine("evaluate");
                    Console.WriteLine("validation perplexity: {0}", Evaluate(model, validData));

                    if (epoch >= 6)
                    {
                        sgd.LearningRate /= 1.2f;
                        Console.WriteLine("learning rate =" + sgd.LearningRate);
                    }
                }
            }

            Console.WriteLine("test start");
            Console.WriteLine("test perplexity:" + Evaluate(model, testData));
        }

        static Real Evaluate(FunctionStack<Real> model, int[] dataset)
        {
            FunctionStack<Real> predictModel = DeepCopyHelper<Real>.DeepCopy(model);
            predictModel.ResetState();

            Real totalLoss = 0;
            long totalLossCount = 0;

            for (int i = 0; i < dataset.Length - 1; i++)
            {
                NdArray<Real> x = new NdArray<Real>(new[] { 1 }, BATCH_SIZE);
                NdArray<int> t = new NdArray<int>(new[] { 1 }, BATCH_SIZE);

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x.Data[j] = dataset[j + i];
                    t.Data[j] = dataset[j + i + 1];
                }

                NdArray<Real> result = predictModel.Forward(x)[0];
                Real sumLoss = new SoftmaxCrossEntropy<Real>().Evaluate(result, t);
                totalLoss += sumLoss;
                totalLossCount++;
            }

            //calc perplexity
            return Math.Exp(totalLoss / (totalLossCount - 1));
        }
    }
}
