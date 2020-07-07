using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.CL;
using KelpNet.Tools;

#if DOUBLE
#elif NETCOREAPP2_0
using Math = System.MathF;
#else
using Math = KelpNet.MathF;
#endif

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    //Rigging the Lottery: Making All Tickets Winners
    //https://arxiv.org/abs/1911.11134
    //https://github.com/google-research/rigl/tree/master/rigl/mnist/mnist_train_eval.py

    class Sample21
    {
        //ミニバッチの数
        const int BATCH_SIZE = 100;

        const int NUM_EPOCHS = 200;

        const int LR_DROP_EPOCH = 75; // The epoch to start dropping lr.

        const Real LEARNING_RATE = 0.2f;

        const Real DROP_FRACTION = 0.3f;       // When changing mask dynamically, this fraction decides how much of the
        const Real SPARSITY_SCALE = 0.9f;      // Relative sparsity of second layer.
        const Real RIGL_ACC_SCALE = 0.0f;      // Used to scale initial accumulated gradients for new connections.

        const int MASKUPDATE_BEGIN_STEP = 0;   // Step to begin mask updates.
        const int MASKUPDATE_END_STEP = 50000; // Step to end mask updates.
        const int MASKUPDATE_FREQUENCY = 100;  // Step interval between mask updates.

        const Real END_SPARSITY = 0.98f;       // desired sparsity of final model. //希望する最終モデルのまばら度

        const Real L2_SCALE = 1e-4f;           //l2 loss scale

        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST data loading...");
            MnistData<Real> mnistData = new MnistData<Real>();

            //テストデータから全データを取得
            TestDataSet<Real> datasetY = mnistData.Eval.GetAllDataSet();

            Console.WriteLine("Network initializing...");

            int numBatches = mnistData.Train.Length / BATCH_SIZE; // 600 = 60000 / 100
            int batchPerEpoch = mnistData.Train.Length / BATCH_SIZE;
            int[] boundaries = { LR_DROP_EPOCH * batchPerEpoch, (LR_DROP_EPOCH + 20) * batchPerEpoch };

            Dictionary<string, Real> customSparsities = new Dictionary<string, Real>
            {
                { "layer2", END_SPARSITY * SPARSITY_SCALE },
                { "layer3", END_SPARSITY * 0 }
            };

            var layer1 = new MaskedLinear<Real>(28 * 28, 300, name: "layer1");
            var layer2 = new MaskedLinear<Real>(300, 100, name: "layer2");
            var layer3 = new Linear<Real>(100, 10, name: "layer3");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack<Real> nn = new FunctionStack<Real>(
                layer1,
                new ReLU<Real>(name: "l1 ReLU"),
                layer2,
                new ReLU<Real>(name: "l2 ReLU"),
                layer3
            );

            SoftmaxCrossEntropy<Real> sce = new SoftmaxCrossEntropy<Real>();

            WeightDecay<Real> weightDecay = new WeightDecay<Real>(L2_SCALE);
            weightDecay.AddParameters(layer1.Weight, layer2.Weight, layer3.Weight);

            MomentumSGD<Real> mSGD = new MomentumSGD<Real>();
            mSGD.SetUp(nn);

            var opt = new SparseRigLOptimizer(mSGD, MASKUPDATE_BEGIN_STEP, MASKUPDATE_END_STEP, MASKUPDATE_FREQUENCY, DROP_FRACTION, "cosine", "zeros", RIGL_ACC_SCALE);

            Real[][] allMasks =
            {
                layer1.Mask.Data,
                layer2.Mask.Data,
            };

            int[][] Shapes =
            {
                layer1.Weight.Shape,
                layer2.Weight.Shape,
            };

            string[] Names =
            {
                layer1.Name,
                layer2.Name,
            };

            NdArray<Real>[] allWights =
            {
                layer1.Weight,
                layer2.Weight,
            };

            //マスクの初期化
            SparseUtils.MaskInit(allMasks, Shapes, Names, "erdos_renyi", END_SPARSITY, customSparsities);

            Console.WriteLine("[Global sparsity] " + SparseUtils.CalculateSparsity(allMasks));
            var weightSparsity = GetWeightSparsity(allMasks);
            Console.WriteLine("[Sparsity] Layer0, Layer1 : " + weightSparsity[0] + ", " + weightSparsity[1]);


            Console.WriteLine("Training Start...");


            //学習開始
            for (int i = 0; i < NUM_EPOCHS * numBatches; i++)
            {
                //訓練データからランダムにデータを取得
                TestDataSet<Real> datasetX = mnistData.Train.GetRandomDataSet(BATCH_SIZE);

                //バッチ学習を実行する
                NdArray<Real> y = nn.Forward(datasetX.Data)[0];
                Real loss = sce.Evaluate(y, datasetX.Label);
                nn.Backward(y);

                weightDecay.Update();
                opt._optimizer.LearningRate = PiecewiseConstant(opt._optimizer.UpdateCount, boundaries, LEARNING_RATE);

                opt.condMaskUpdate(allMasks, allWights);

                //20回に1回結果出力
                if (i % 10 + 1 == 10)
                {
                    Console.WriteLine("\nbatch count:" + (i + 1) + " (lr:" + opt._optimizer.LearningRate + ")");
                    Console.WriteLine("loss " + loss);
                }

                //精度をテストする
                if (i % numBatches + 1 == numBatches)
                {
                    Console.WriteLine("\nTesting...");

                    //テストを実行
                    Real accuracy = Trainer.Accuracy(nn, datasetY, new SoftmaxCrossEntropy<Real>(), out loss);

                    Console.WriteLine("Epoch, Iteration, loss, accuracy");
                    Console.WriteLine(Math.Floor((i + 1) / (Real)numBatches) + ", " + (i + 1) + ", " + loss + ", " + accuracy);
                }
            }
        }

        static Real PiecewiseConstant(long globalCount, int[] boundaries, Real baseValue)
        {
            Real result = baseValue;

            for (int i = 0; i < boundaries.Length; i++)
            {
                if (boundaries[i] < globalCount)
                {
                    result = baseValue / Math.Pow(3.0f, i + 1);
                }
            }

            return result;
        }

        static Real[] GetWeightSparsity(Real[][] mask)
        {
            Real[] result = new Real[mask.Length];

            for (int i = 0; i < mask.Length; i++)
            {
                result[i] = 1.0f - mask[i].Sum() / mask[i].Length;
            }

            return result;
        }
    }

    class SparseRigLOptimizer : SparseSETOptimizer
    {
        private Real _initialAccScale;

        public SparseRigLOptimizer(MomentumOptimizer<Real> optimizer, int beginStep, int endStep, int frequency, Real dropFraction = 0.1f, string dropFractionAnneal = "constant", string growInit = "zeros", Real initialAccScale = 0.0f) : base(optimizer, beginStep, endStep, frequency, dropFraction, dropFractionAnneal, growInit)
        {
            _initialAccScale = initialAccScale;
        }

        public override void genericMaskUpdate(Real[] mask, NdArray<Real> weight)
        {
            Real[] scoreDrop = new Real[weight.Length];
            Real[] scoreGrow = new Real[weight.Length];

            for (int i = 0; i < scoreDrop.Length; i++)
            {
                scoreDrop[i] = Math.Abs(mask[i] * weight.Data[i]);//マスク前の重みにマスクを掛ける　元の実装だとここに1e-5の正規乱数が足される
                scoreGrow[i] = mask[i] * weight.Grad[i];//gradはマスク済みの重みに行われた値
            }

            //マスクと重みを更新
            Update(scoreDrop, scoreGrow, mask, weight);
        }

        //更新のあった値のモーメンタムをリセットする
        public override void ResetMomentum(bool[] newConnections, Real[] mask, NdArray<Real> weight)
        {
            for (int i = 0; i < _optimizer.FunctionParameters.Count; i++)
            {
                if (_optimizer.FunctionParameters[i].Name == weight.Name)
                {
                    for (int j = 0; j < newConnections.Length; j++)
                    {
                        if (newConnections[j])
                        {
                            _optimizer.var[i][j] = mask[j] * weight.Grad[j] * _initialAccScale;
                        }
                    }
                }
            }
        }
    }

    class SparseSETOptimizer
    {
        public MomentumOptimizer<Real> _optimizer;
        private Real _dropFractionInitialValue; // of connections to drop during each update. //ドロップ率:各更新時にドロップする接続の割合
        private int _beginStep; // first iteration where masks are updated.
        private int _endStep; // iteration after which no mask is updated.
        private int _frequency; // of mask update operations.
        private int _frequencyVal;
        private Real _dropFraction;
        private long lastUpdateStep;

        protected SparseSETOptimizer(MomentumOptimizer<Real> optimizer, int beginStep, int endStep, int frequency, Real dropFraction = 0.1f, string dropFractionAnneal = "constant", string growInit = "zeros")
        {
            this._optimizer = optimizer;
            this._dropFractionInitialValue = dropFraction;
            this._beginStep = beginStep;
            this._endStep = endStep;
            this._frequency = frequency;
            this._frequencyVal = frequency;
            lastUpdateStep = -_frequencyVal;
        }

        public bool isMaskUpdateIter(long globalStep)
        {
            int beginStep = _beginStep;
            int endStep = _endStep;
            int frequency = _frequency;

            bool isStepWithinUpdateRange = globalStep >= beginStep && (globalStep <= endStep || endStep < 0);
            bool isUpdateStep = lastUpdateStep + frequency <= globalStep;
            bool isMaskUpdateIter = isStepWithinUpdateRange && isUpdateStep;

            _dropFraction = getDropFraction(globalStep, isMaskUpdateIter);

            return isMaskUpdateIter;
        }

        Real getDropFraction(long globalStep, bool isMaskUpdateIter, Real alpha = 0.0f)
        {
            int decaySteps = _endStep - _beginStep;

            long step = Math.Min(globalStep, decaySteps);
            Real cosine_decay = 0.5f * (1.0f + Math.Cos(Math.PI * step / decaySteps));
            Real decayed = (1 - alpha) * cosine_decay + alpha;

            if (isMaskUpdateIter)
            {
                return _dropFractionInitialValue * decayed;
            }
            else
            {
                return 0.0f;
            }
        }

        public void condMaskUpdate(Real[][] masks, NdArray<Real>[] weights)
        {
            long globalStep = _optimizer.UpdateCount;

            if (isMaskUpdateIter(globalStep))
            {
                for (int i = 0; i < masks.Length; i++)
                {
                    genericMaskUpdate(masks[i], weights[i]);
                }

                lastUpdateStep = globalStep;
            }
            else
            {
                _optimizer.Update();
            }

        }

        public virtual void genericMaskUpdate(Real[] mask, NdArray<Real> weight)
        {
        }

        public void Update(Real[] scoreDrop, Real[] scoreGrow, Real[] mask, NdArray<Real> weight)
        {
            int nOnes = (int)mask.Sum();
            int nPrune = (int)Math.Floor(nOnes * _dropFraction);//元実装はキャストで切り捨てしている
            int nKeep = nOnes - nPrune;

            int[] keepMask = new int[mask.Length];
            Real chackval = scoreDrop.OrderBy(a => -a).ElementAt(nKeep);

            //mask1にScoreDropの大きい順の先頭nKeep個に1を入れる
            for (int i = 0, keepCount = 0; i < keepMask.Length && keepCount < nKeep; i++)
            {
                if (scoreDrop[i] >= chackval)
                {
                    keepMask[i] = 1;
                    keepCount++;
                }
            }

            Real[] scoreGrowLifted = new Real[scoreGrow.Length];

            //有効になっている接続のスコアが最も低くなるようにする //元実装だとscoreGrow.Min() - 1.0f
            for (int i = 0; i < keepMask.Length; i++)
            {
                scoreGrowLifted[i] = keepMask[i] == 1 ? Real.MinValue : scoreGrow[i];
            }

            int[] growMask = new int[scoreGrowLifted.Length];

            chackval = scoreGrowLifted.OrderBy(a => -a).ElementAt(nPrune);

            //mask2にscoreGrowLiftedの大きい順の先頭nPrune個に1を入れる
            for (int i = 0, prunedCount = 0; i < growMask.Length && prunedCount < nPrune; i++)
            {
                if (scoreGrowLifted[i] > chackval)
                {
                    growMask[i] = 1;
                    prunedCount++;
                }
            }

            //mask1 * mask2で全て0になるか確認
            int sum = 0;
            for (int i = 0; i < keepMask.Length; i++)
            {
                sum += keepMask[i] * growMask[i];
            }

            if (sum != 0) throw new Exception();

            //int[] glowTensor = getGlowTensor() //元実装がデフォルトだと0固定なので省略

            bool[] newConnections = new bool[weight.Length];

            for (int i = 0; i < weight.Length; i++)
            {
                newConnections[i] = growMask[i] == 1 && mask[i] == 0;

                //マスクの更新があるか判定
                if (newConnections[i])
                {
                    weight.Data[i] = 0; // この0は本来はgetGlowTensor()で取得してきたglowTensor;
                }
            }

            ResetMomentum(newConnections, mask, weight);

            for (int i = 0; i < mask.Length; i++)
            {
                //mask1 * mask2で全て0になる保証があるので
                mask[i] = keepMask[i] + growMask[i];
            }
        }

        public virtual void ResetMomentum(bool[] newConnections, Real[] mask, NdArray<Real> weight)
        {
        }
    }

    class SparseUtils
    {
        public static Real CalculateSparsity(Real[][] allMasks) //allMasks内部は0または1のintだがSum関数のためにReal型
        {
            Real denseParams = 0.0f;
            Real sparseParams = 0.0f;

            for (int i = 0; i < allMasks.Length; i++)
            {
                denseParams += allMasks[i].Length;
                sparseParams += allMasks[i].Sum();
            }

            return 1.0f - sparseParams / denseParams;
        }

        public static void MaskInit(Real[][] allMasks, int[][] Shapes, string[] Names, string method, Real defaultSparsity, Dictionary<string, Real> customSparsityMap)
        {
            Real[] sparsities = { };

            if (method == "erdos_renyi")
            {
                sparsities = GetSparsities(allMasks, Shapes, Names, method, defaultSparsity, customSparsityMap);
            }

            for (int i = 0; i < allMasks.Length; i++)
            {
                GetMaskRandom(allMasks[i], sparsities[i]);
            }
        }

        public static void GetMaskRandom(Real[] mask, Real sparsity)
        {
            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] = 1;
            }

            int nZeros = Math.Ceiling(mask.Length * sparsity);

            for (int i = 0; i < nZeros; i++)
            {
                int index;

                do
                {
                    index = Mother.Dice.Next(mask.Length);
                }
                while (mask[index] == 0);

                mask[index] = 0;
            }
        }

        //各レイヤの粗度を取得する
        public static Real[] GetSparsities(Real[][] allMasks, int[][] Shapes, string[] Names, string method, Real defaultSparsity, Dictionary<string, Real> customSparsityMap)
        {
            //元実装だとMapに粗度を必要とするレイヤが有るかチェックが有る

            return GetSparsitiesErdosRenyi(allMasks, Shapes, Names, defaultSparsity, customSparsityMap);
        }

        public static Real[] GetSparsitiesErdosRenyi(Real[][] allMasks, int[][] Shapes, string[] Names, Real defaultSparsity, Dictionary<string, Real> customSparsityMap)
        {
            // We have to enforce custom sparsities and then find the correct scaling factor.

            bool isEpsValid = false;
            // The following loop will terminate worst case when all masks are in the custom_sparsity_map. 
            // This should probably never happen though, since once we have a single variable or more with the same constant, we have a valid epsilon.
            // Note that for each iteration we add at least one variable to the custom_sparsity_map and therefore this while loop should terminate.
            List<int> denseLayerIndex = new List<int>();

            // We will start with all layers and try to find right epsilon. 
            // However if any probablity exceeds 1, we will make that layer dense and repeat the process (finding epsilon) with the non-dense layers.
            // We want the total number of connections to be the same. 
            // Let say we have for layers with N_1, ..., N_4 parameters each.
            // Let say after some iterations probability of some dense layers (3, 4) exceeded 1 and therefore we added them to the dense_layers set.
            // Those layers will not scale with erdos_renyi, however we need to count them so that target paratemeter count is achieved.
            // See below.
            // eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) = (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            // eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            // eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
            Real[] rawProbabilities = new Real[allMasks.Length];
            Real eps = 0;

            while (!isEpsValid)
            {
                Real divisor = 0;
                int rhs = 0;

                for (int i = 0; i < allMasks.Length; i++)
                {
                    int nParam = allMasks[i].Length;
                    int nZeros = Math.Ceiling(nParam * defaultSparsity);

                    if (denseLayerIndex.Contains(i))
                    {
                        //See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= nZeros;
                    }
                    else if (customSparsityMap.ContainsKey(Names[i]))
                    {
                        //We ignore custom_sparsities in erdos-renyi calculations.
                        continue;
                    }
                    else
                    {
                        //Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the equation above.
                        int nOnes = nParam - nZeros;
                        rhs += nOnes;

                        rawProbabilities[i] = (Shapes[i][0] + Shapes[i][1]) / (Real)allMasks[i].Length;
                    }

                    divisor += rawProbabilities[i] * nParam;
                }

                // By multipliying individual probabilites with epsilon, we should get the number of parameters per layer correctly.
                eps = rhs / divisor;

                // If eps * raw_probabilities[mask.name] > 1.0 We set the sparsities of that mask to 0.0, so they become part of dense_layers sets.
                Real maxProb = rawProbabilities.Max();
                Real maxProbOne = maxProb * eps;

                if (maxProbOne > 1)
                {
                    isEpsValid = false;

                    for (int i = 0; i < rawProbabilities.Length; i++)
                    {
                        if (rawProbabilities[i] == maxProb)
                        {
                            denseLayerIndex.Add(i);
                        }
                    }
                }
                else
                {
                    isEpsValid = true;
                }
            }

            Real[] sparsities = new Real[allMasks.Length];

            for (int i = 0; i < allMasks.Length; i++)
            {
                if (customSparsityMap.ContainsKey(Names[i]))
                {
                    sparsities[i] = customSparsityMap[Names[i]];
                }
                else if (denseLayerIndex.Contains(i))
                {
                    sparsities[i] = 0.0f;
                }
                else
                {
                    sparsities[i] = 1.0f - eps * rawProbabilities[i];
                }
            }

            return sparsities;
        }
    }
}
