using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace CaffemodelLoader
{
    [Serializable]
    public class FunctionDictionary : Function
    {
        const string FUNCTION_NAME = "FunctionDictionary";

        //出力名称のKey付きのファンクションブロック
        public Dictionary<string, FunctionSet> FunctionBlocks = new Dictionary<string, FunctionSet>();

        //辞書の実行順リスト
        public List<string> FunctionBlockNames = new List<string>();

        //コンストラクタ
        public FunctionDictionary(string name = FUNCTION_NAME) : base(name)
        {
        }

        public void Add(string functionBlockName, Function function, string[] inputNames)
        {
            if (FunctionBlocks.ContainsKey(functionBlockName))
            {
                //追加
                FunctionBlocks[functionBlockName].Add(function);
            }
            else
            {
                //新規
                FunctionBlockNames.Add(functionBlockName);
                FunctionSet functionSet = new FunctionSet(function, inputNames);
                FunctionBlocks.Add(functionBlockName, functionSet);
            }
        }

        //Forward
        public override NdArray[] Forward(params NdArray[] xs)
        {
            NdArray[] result =  FunctionBlocks[FunctionBlockNames[0]].Forward(xs);

            for (int i = 1; i < FunctionBlockNames.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[FunctionBlockNames[i]].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.AddRange(FunctionBlocks[inputBlockNames[j]].Result);
                }

                result = FunctionBlocks[FunctionBlockNames[i]].Forward(inputData.ToArray());
            }

            return result;
        }

        //Backward
        public override void Backward(params NdArray[] y)
        {
            NdArray.Backward(y[0]);
        }

        //重みの更新処理
        public override void Update()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Value.Update();
            }
        }

        //ある処理実行後に特定のデータを初期値に戻す処理
        public override void ResetState()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Value.ResetState();
            }
        }

        //予想を実行する
        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] result = FunctionBlocks[FunctionBlockNames[0]].Predict(xs);

            for (int i = 1; i < FunctionBlockNames.Count; i++)
            {
                string[] inputBlockNames = FunctionBlocks[FunctionBlockNames[i]].InputNames;
                List<NdArray> inputData = new List<NdArray>();

                for (int j = 0; j < inputBlockNames.Length; j++)
                {
                    inputData.AddRange(FunctionBlocks[inputBlockNames[j]].Result);
                }

                result = FunctionBlocks[FunctionBlockNames[i]].Predict(inputData.ToArray());
            }

            return result;
        }

        public override void SetOptimizer(params Optimizer[] optimizers)
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Value.SetOptimizer(optimizers);
            }
        }
    }
}
