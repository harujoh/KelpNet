using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using CIFARLoader.Common;

namespace CIFARLoader
{
    public class Tar
    {
        public static Dictionary<string, byte[]> GetExtractedStreams(string archive)
        {
            Dictionary<string, byte[]> result = new Dictionary<string, byte[]>();

            byte[] block = new byte[512];

            RawSerializer<HeaderBlock> serializer = new RawSerializer<HeaderBlock>();

            using (Stream fs = new GZipStream(File.Open(archive, FileMode.Open, FileAccess.Read), CompressionMode.Decompress, false))
            {
                while (fs.Read(block, 0, block.Length) > 0)
                {
                    HeaderBlock hb = serializer.RawDeserialize(block);

                    var remainingBytes = hb.GetSize();

                    if (hb.typeflag == 0)
                    {
                        hb.typeflag = (byte)'0';
                    }

                    var blocksToMunch = remainingBytes > 0 ? (remainingBytes - 1) / 512 + 1 : 0;

                    if ((TarEntryType)hb.typeflag == TarEntryType.File_Old ||
                        (TarEntryType)hb.typeflag == TarEntryType.File ||
                        (TarEntryType)hb.typeflag == TarEntryType.File_Contiguous)
                    {
                        List<byte> output = new List<byte>();

                        while (blocksToMunch > 0)
                        {
                            int bytesToWrite = block.Length < remainingBytes ? block.Length : remainingBytes;

                            byte[] tmpArray = new byte[bytesToWrite];
                            Array.Copy(block, 0, tmpArray, 0, bytesToWrite);

                            output.AddRange(tmpArray);

                            remainingBytes -= bytesToWrite;

                            blocksToMunch--;

                            if (fs.Read(block, 0, block.Length) > 0) break;
                        }

                        result.Add(hb.GetName(), output.ToArray());
                    }
                }
            }

            return result;
        }
    }
}
