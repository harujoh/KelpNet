using System;
using System.Diagnostics;

namespace KelpNet.Benchmark
{
    class CpuBenchmarker
    {
        //VGG16のLinearの最大メモリを想定
        const int INPUT_SIZE = 25088;
        const int OUTPUT_SIZE = 4096;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();


            //1次元データを用意
            NdArray inputArrayCpu = new NdArray(BenchDataMaker.GetFloatArray(INPUT_SIZE));
            NdArray inputArrayGpu = new NdArray(BenchDataMaker.GetFloatArray(INPUT_SIZE));


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


            //2次元3チャンネルのデータを用意
            NdArray inputImageArrayGpu = new NdArray(BenchDataMaker.GetFloatArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);
            NdArray inputImageArrayCpu = new NdArray(BenchDataMaker.GetFloatArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);


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
        }
    }
}
