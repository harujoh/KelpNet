using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace KelpNet.Tools
{
    public class Tar
    {
        public static Dictionary<string, byte[]> GetExtractedStreams(string archive)
        {
            Dictionary<string, byte[]> result = new Dictionary<string, byte[]>();

            byte[] block = new byte[512];
            RawSerializer<HeaderBlock> serializer = new RawSerializer<HeaderBlock>();
            int blocksToMunch = 0;
            int remainingBytes = 0;

            bool outputFlg = false;
            string name = String.Empty;

            List<byte> output = new List<byte>();

            using (Stream fs = new GZipStream(File.Open(archive, FileMode.Open, FileAccess.Read), CompressionMode.Decompress, false))
            {
                while (fs.Read(block, 0, block.Length) > 0)
                {
                    if (blocksToMunch > 0)
                    {
                        if (outputFlg)
                        {
                            int bytesToWrite = block.Length < remainingBytes ? block.Length : remainingBytes;

                            byte[] tmpArray = new byte[bytesToWrite];
                            Array.Copy(block, 0, tmpArray, 0, bytesToWrite);

                            output.AddRange(tmpArray);

                            remainingBytes -= bytesToWrite;
                        }

                        blocksToMunch--;

                        if (blocksToMunch == 0)
                        {
                            outputFlg = false;
                        }

                        continue;
                    }

                    if (name != string.Empty)
                    {
                        result.Add(name, output.ToArray());
                        output.Clear();
                    }

                    HeaderBlock hb = serializer.RawDeserialize(block);


                    if (name == String.Empty || (TarEntryType)hb.typeflag != TarEntryType.GnuLongName)
                    {
                        name = hb.GetName();
                    }

                    if (name == String.Empty) break;
                    remainingBytes = hb.GetSize();

                    if (hb.typeflag == 0)
                    {
                        hb.typeflag = (byte)'0';
                    }

                    blocksToMunch = remainingBytes > 0 ? (remainingBytes - 1) / 512 + 1 : 0;

                    if ((TarEntryType)hb.typeflag == TarEntryType.File_Old ||
                        (TarEntryType)hb.typeflag == TarEntryType.File ||
                        (TarEntryType)hb.typeflag == TarEntryType.File_Contiguous)
                    {
                        outputFlg = true;
                    }
                }
            }

            return result;
        }
    }
}
