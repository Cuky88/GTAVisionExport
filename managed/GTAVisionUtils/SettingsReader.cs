using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitMiracle.LibTiff.Classic;
using IniParser;
using IniParser.Model;

namespace GTAVisionUtils {

    public class SettingsReader
    {
        public const float maxAnnotationRange = 150f; //150f;
        public const string DATA_FOLDER = @"D:\Devel\GTAVisionExport\managed\Data\";
        public const bool DEBUG_TRANS = false;
        public const string DEBUG_PATH = @"D:\Devel\GTAVisionExport\managed\Data\transformation.txt";
        public const int runLoop = 20000;
        public const bool getPeds = false;
    }
}
