using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;

namespace KelpNetTester.Benchmarker
{
    class SingleBenchmark
    {
        //VGG16のLinearの最大メモリを想定
        const int INPUT_SIZE = 25088;
        const int OUTPUT_SIZE = 4096;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();

            NdArray inputArrayCpu = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));
            NdArray inputArrayGpu = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));

            //Linear
            Linear linear = new Linear(INPUT_SIZE, OUTPUT_SIZE);
            Console.WriteLine("◆" + linear.Name);

            sw.Restart();
            NdArray[] gradArrayCpu = linear.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data; //DataをGradとして使用

            sw.Restart();
            linear.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (linear.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = linear.Forward(inputArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                linear.Backward(gradArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }


            //Tanh
            Tanh tanh = new Tanh();
            Console.WriteLine("\n◆" + tanh.Name);

            sw.Restart();
            gradArrayCpu = tanh.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            tanh.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (tanh.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = tanh.Forward(inputArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                tanh.Backward(gradArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }


            //Sigmoid
            Sigmoid sigmoid = new Sigmoid();
            Console.WriteLine("\n◆" + sigmoid.Name);

            sw.Restart();
            gradArrayCpu = sigmoid.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sigmoid.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (sigmoid.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = sigmoid.Forward(inputArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                sigmoid.Backward(gradArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }


            //ReLU
            ReLU relu = new ReLU();
            Console.WriteLine("\n◆" + relu.Name);

            sw.Restart();
            gradArrayCpu = relu.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            relu.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (relu.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = relu.Forward(inputArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                relu.Backward(gradArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }


            //LeakyReLU
            LeakyReLU leakyRelu = new LeakyReLU();
            Console.WriteLine("\n◆" + leakyRelu.Name);

            sw.Restart();
            gradArrayCpu = leakyRelu.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            leakyRelu.Backward(gradArrayCpu);
            sw.Stop();

            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (leakyRelu.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = leakyRelu.Forward(inputArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                leakyRelu.Backward(gradArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }


            NdArray inputImageArrayGpu = new NdArray(BenchDataMaker.GetRealArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);
            NdArray inputImageArrayCpu = new NdArray(BenchDataMaker.GetRealArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);


            //MaxPooling
            MaxPooling maxPooling = new MaxPooling(3);
            Console.WriteLine("\n◆" + maxPooling.Name);

            sw.Restart();
            NdArray[] gradImageArrayCpu = maxPooling.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            maxPooling.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (maxPooling.SetGpuEnable(true))
            {
                sw.Restart();
                maxPooling.Forward(inputImageArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                //メモリ転送のみのため実装がない
                Console.WriteLine("Backward[Gpu] : None");
            }


            //Conv2D
            Convolution2D conv2d = new Convolution2D(3, 3, 3);
            Console.WriteLine("\n◆" + conv2d.Name);

            sw.Restart();
            gradImageArrayCpu = conv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            conv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (conv2d.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradImageArrayGpu = conv2d.Forward(inputImageArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradImageArrayGpu[0].Grad = gradImageArrayGpu[0].Data;

                sw.Restart();
                conv2d.Backward(gradImageArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }


            //Deconv2D
            Deconvolution2D deconv2d = new Deconvolution2D(3, 3, 3);
            Console.WriteLine("\n◆" + deconv2d.Name);

            sw.Restart();
            gradImageArrayCpu = deconv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            deconv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (deconv2d.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradImageArrayGpu = deconv2d.Forward(inputImageArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradImageArrayGpu[0].Grad = gradImageArrayGpu[0].Data;

                sw.Restart();
                deconv2d.Backward(gradImageArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }

            //Dropout
            Dropout dropout = new Dropout();
            Console.WriteLine("\n◆" + dropout.Name);

            sw.Restart();
            gradArrayCpu = dropout.Forward(inputArrayCpu);
            sw.Stop();
            Console.WriteLine("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            dropout.Backward(gradArrayCpu);
            sw.Stop();
            Console.WriteLine("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (dropout.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = dropout.Forward(inputArrayGpu);
                sw.Stop();
                Console.WriteLine("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                dropout.Backward(gradArrayGpu);
                sw.Stop();
                Console.WriteLine("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
        }
    }
}
