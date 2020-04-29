using System;

namespace KelpNet
{
    //乱数の素
    //C#ではRandomを複数同時にインスタンスすると似たような値しか吐かないため
    //一箇所でまとめて管理しておく必要がある
    public class Mother  // Der Alte würfelt nicht.
    {
#if DEBUG
        //デバッグ時はシードを固定
        public static Random Dice = new Random(128);
#else
        public static Random Dice = new Random();
#endif
    }
}
