using System;
using System.IO;
using System.Net;
using System.Threading.Tasks;

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
                Task task = Task.Factory.StartNew(() => downloadClient.DownloadFile(new Uri(url), savedPath));
                task.Wait();
            }

            return savedPath;
        }

    }
}
