using System;
using System.IO;
using System.Net;
using System.Security.Cryptography;

namespace KelpNet.Tools
{
    public class InternetFileDownloader
    {
        private const string TMP_DATA_PATH = "KelpNet/TestData/";
        static readonly string TmpFolderPath = Path.Combine(Path.GetTempPath(), TMP_DATA_PATH);

        public static string Donwload(string url, string fileName, string hash = "", string saveFolder = null)
        {
            WebClient downloadClient = new WebClient();

            string savedPath = Path.Combine(saveFolder ?? TmpFolderPath, fileName);

            if (saveFolder == null && File.Exists(fileName))
            {
                if (LoadFile(fileName, hash))
                {
                    return fileName;
                }

                //前回ダウンロードに失敗している
                Console.WriteLine(fileName + "は破損しているためダウンロードを行います");
            }

            if (File.Exists(savedPath))
            {
                if (LoadFile(savedPath, hash))
                {
                    return savedPath;
                }

                //前回ダウンロードに失敗している
                Console.WriteLine(fileName + "が破損しているため再ダウンロードを行います");

                //失敗していたファイルを削除
                File.Delete(savedPath);
            }

            //ファイルのチェックとダウンロード
            Console.WriteLine(fileName + "をダウンロードします");

            //テンポラリにフォルダがなければ作成する
            if (!Directory.Exists(TmpFolderPath))
            {
                Directory.CreateDirectory(TmpFolderPath);
            }

            //非同期ダウンロードを開始する
            downloadClient.DownloadFileTaskAsync(new Uri(url), savedPath).Wait();

            return savedPath;
        }

        static bool LoadFile(string path, string hash)
        {
            //ファイルを開く
            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (hash != string.Empty)
                {
                    //ハッシュ値を計算する
                    byte[] bs = new MD5CryptoServiceProvider().ComputeHash(fs);

                    return BitConverter.ToString(bs).ToLower().Replace("-", "") == hash;
                }

                return true;
            }
        }
    }
}
