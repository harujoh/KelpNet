using System;
using System.IO;
using System.Net;

namespace TestDataManager
{
    public class InternetFileDownloader
    {
        private const string TMP_DATA_PATH = "KelpNet/TestData/";
        static readonly string TmpFolderPath = Path.Combine(Path.GetTempPath(), TMP_DATA_PATH);

        public static string Donwload(string url, string fileName)
        {
            WebClient downloadClient = new WebClient();

            string savedPath = Path.Combine(TmpFolderPath, fileName);

            if (File.Exists(fileName))
            {
                return fileName;
            }

            //ファイルのチェックとダウンロード
            if (!File.Exists(savedPath))
            {
                Console.WriteLine(fileName + "をダウンロードします");

                if (!Directory.Exists(TmpFolderPath))
                {
                    Directory.CreateDirectory(TmpFolderPath);
                }

                //非同期ダウンロードを開始する
                downloadClient.DownloadFileTaskAsync(new Uri(url), savedPath).Wait();
            }

            return savedPath;
        }
    }
}
