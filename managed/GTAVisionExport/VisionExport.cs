using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows.Forms;
using GTA;
using GTA.Math;
using System.Drawing;
using Amazon.S3;
using System.Net;
using VAutodrive;
using System.Net.Sockets;
using GTAVisionUtils;
using System.IO.Compression;
using GTA.Native;
using IniParser;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;
using System.Runtime.Serialization.Formatters.Binary;

namespace GTAVisionExport
{
    [JsonObject(IsReference = true)]
    public class CollectedData
    {
        public string Image { get; set; }
        public DateTime RealTime { get; set; }
        public TimeSpan GameTime { get; set; }
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public int UIwidth { get; set; }
        public int UIheight { get; set; }
        public int CamHash { get; set; }
        public int GameTime2 { get; set; }
        public double CamFOV { get; set; }
        public double CamNearClip { get; set; }
        public double CamFarClip { get; set; }
        public GTAVector Campos { get; set; }
        public GTAVector Camrot { get; set; }
        public GTAVector Camdir { get; set; }
        public GTAVector Carpos { get; set; }
        public GTAVector Carrot { get; set; }
        public DenseMatrix PMatrix { get; set; }
        public DenseMatrix VMatrix { get; set; }
        public GTAVector Gamerpos { get; set; }
        public List<GTADetection> Detections = new List<GTADetection>();
    }

    class VisionExport : Script
    {
#if DEBUG
        const string session_name = "NEW_DATA_CAPTURE_NATURAL_V4_3";
#else
        const string session_name = "NEW_DATA_CAPTURE_NATURAL_V4_3";
#endif
        //private readonly string dataPath =
        //    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Data");
        private readonly string dataPath = @"D:\Devel\GTAVisionExport\managed\Data\";
        private readonly Weather[] wantedWeather = new Weather[] { Weather.Clear, Weather.ExtraSunny, Weather.Overcast, Weather.Foggy, Weather.Clouds };
        private Player player;
        private string outputPath;
        private GTARun run;
        private bool enabled = false;
        private Socket server;
        private Socket connection;
        private UTF8Encoding encoding = new UTF8Encoding(false);
        private KeyHandling kh = new KeyHandling();
        private ZipArchive archive;
        private Stream S3Stream;
        private AmazonS3Client client;
        private Task postgresTask;
        private Task runTask;
        private int curSessionId = -1;
        private speedAndTime lowSpeedTime = new speedAndTime();
        private bool IsGamePaused = false;
        private StereoCamera cams;
        private string path;
        private int FileCounter = 1;

        public VisionExport()
        {
            if (!Directory.Exists(dataPath)) Directory.CreateDirectory(dataPath);
            PostgresExport.InitSQLTypes();
            player = Game.Player;
            server = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            server.Bind(new IPEndPoint(IPAddress.Loopback, 5555));
            server.Listen(5);
            //server = new UdpClient(5555);
            var parser = new FileIniDataParser();
            var location = AppDomain.CurrentDomain.BaseDirectory;
            //var data = parser.ReadFile(Path.Combine(location, "GTAVision.ini"));
            var data = new IniParser.Model.IniData();
            var access_key = data["aws"]["access_key"];
            var secret_key = data["aws"]["secret_key"];
            //client = new AmazonS3Client(new BasicAWSCredentials(access_key, secret_key), RegionEndpoint.USEast1);
            //outputPath = @"D:\Datasets\GTA\";
            //outputPath = Path.Combine(outputPath, "testData.yaml");
            //outStream = File.CreateText(outputPath);
            this.Tick += new EventHandler(this.OnTick);
            this.KeyDown += OnKeyDown;

            Interval = 0;
        }

        public void OnTick(object o, EventArgs e)
        {

        }

        public Bitmap CaptureScreen()
        {
            var cap = new Bitmap(Screen.PrimaryScreen.Bounds.Width, Screen.PrimaryScreen.Bounds.Height);
            var gfx = Graphics.FromImage(cap);
            //var dat = GTAData.DumpData(Game.GameTime + ".jpg");
            gfx.CopyFromScreen(0, 0, 0, 0, cap.Size);
            /*
            foreach (var ped in dat.ClosestPeds) {
                var w = ped.ScreenBBMax.X - ped.ScreenBBMin.X;
                var h = ped.ScreenBBMax.Y - ped.ScreenBBMin.Y;
                var x = ped.ScreenBBMin.X;
                var y = ped.ScreenBBMin.Y;
                w *= cap.Size.Width;
                h *= cap.Size.Height;
                x *= cap.Size.Width;
                y *= cap.Size.Height;
                gfx.DrawRectangle(new Pen(Color.Lime), x, y, w, h);
            } */
            return cap;
            //cap.Save(GetFileName(".png"), ImageFormat.Png);

        }

        public void EnterVehicle()
        {
            /*
            var vehicle = World.GetClosestVehicle(player.Character.Position, 30f);
            player.Character.SetIntoVehicle(vehicle, VehicleSeat.Driver);
            */
            Model mod = new Model(GTA.Native.VehicleHash.Asea);
            var vehicle = GTA.World.CreateVehicle(mod, player.Character.Position);
            player.Character.SetIntoVehicle(vehicle, VehicleSeat.Driver);
            //vehicle.Alpha = 0; //transparent
            //player.Character.Alpha = 0;
        }

        public void ToggleNavigation()
        {
            //YOLO
            MethodInfo inf = kh.GetType().GetMethod("AtToggleAutopilot", BindingFlags.NonPublic | BindingFlags.Instance);
            inf.Invoke(kh, new object[] { new KeyEventArgs(Keys.J) });
        }

        public void ReloadGame()
        {
            /*
            Process p = Process.GetProcessesByName("Grand Theft Auto V").FirstOrDefault();
            if (p != null)
            {
                IntPtr h = p.MainWindowHandle;
                SetForegroundWindow(h);
                SendKeys.SendWait("{ESC}");
                //Script.Wait(200);
            }
            */
            // or use CLEAR_AREA_OF_VEHICLES
            Ped player = Game.Player.Character;
            //UI.Notify("x = " + player.Position.X + "y = " + player.Position.Y + "z = " + player.Position.Z);
            // no need to release the autodrive here
            // delete all surrounding vehicles & the driver's car
            Function.Call(GTA.Native.Hash.CLEAR_AREA_OF_VEHICLES, player.Position.X, player.Position.Y, player.Position.Z, 1000f, false, false, false, false);
            player.LastVehicle.Delete();
            // teleport to the spawning position, defined in GameUtils.cs, subject to changes
            player.Position = GTAConst.StartPos;
            Function.Call(GTA.Native.Hash.CLEAR_AREA_OF_VEHICLES, player.Position.X, player.Position.Y, player.Position.Z, 100f, false, false, false, false);
            // start a new run
            EnterVehicle();
            //Script.Wait(2000);
            ToggleNavigation();

            lowSpeedTime.clearTime();

        }

        public void TraverseWeather()
        {
            for (int i = 1; i < 14; i++)
            {
                //World.Weather = (Weather)i;
                World.TransitionToWeather((Weather)i, 0.0f);
                //Script.Wait(1000);
            }
        }

        public void OnKeyDown(object o, KeyEventArgs k)
        {
            //if (k.KeyCode == Keys.PageUp)
            //{
            //    postgresTask?.Wait();
            //    postgresTask = StartSession();
            //    runTask?.Wait();
            //    runTask = StartRun();
            //    UI.Notify("GTA Vision Enabled");
            //}
            //if (k.KeyCode == Keys.PageDown)
            //{
            //    StopRun();
            //    StopSession();
            //    UI.Notify("GTA Vision Disabled");
            //}
            if (k.KeyCode == Keys.H) // temp modification
            {
                EnterVehicle();
                UI.Notify("Trying to enter vehicle");
                ToggleNavigation();
            }
            if (k.KeyCode == Keys.Y) // temp modification
            {
                ReloadGame();
            }
            //if (k.KeyCode == Keys.U) // temp modification
            //{
            //    var settings = ScriptSettings.Load("GTAVisionExport.xml");
            //    var loc = AppDomain.CurrentDomain.BaseDirectory;

            //    //UI.Notify(ConfigurationManager.AppSettings["database_connection"]);
            //    var str = settings.GetValue("", "ConnectionString");
            //    UI.Notify(loc);

            //}
            if (k.KeyCode == Keys.G) // temp modification
            {
                /*
                IsGamePaused = true;
                Game.Pause(true);
                Script.Wait(500);
                TraverseWeather();
                Script.Wait(500);
                IsGamePaused = false;
                Game.Pause(false);
                */
                var data = GTAData.DumpData(Game.GameTime + ".tiff", new List<Weather>(wantedWeather));

                string path = @"D:\Devel\GTAVisionExport\managed\Data\trymatrix.txt";
                // This text is added only once to the file.
                if (!File.Exists(path))
                {
                    // Create a file to write to.
                    using (StreamWriter file = File.CreateText(path))
                    {


                        file.WriteLine("cam direction file");
                        file.WriteLine("direction:");
                        file.WriteLine(GameplayCamera.Direction.X.ToString() + ' ' + GameplayCamera.Direction.Y.ToString() + ' ' + GameplayCamera.Direction.Z.ToString());
                        file.WriteLine("Dot Product:");
                        file.WriteLine(Vector3.Dot(GameplayCamera.Direction, GameplayCamera.Rotation));
                        file.WriteLine("position:");
                        file.WriteLine(GameplayCamera.Position.X.ToString() + ' ' + GameplayCamera.Position.Y.ToString() + ' ' + GameplayCamera.Position.Z.ToString());
                        file.WriteLine("rotation:");
                        file.WriteLine(GameplayCamera.Rotation.X.ToString() + ' ' + GameplayCamera.Rotation.Y.ToString() + ' ' + GameplayCamera.Rotation.Z.ToString());
                        file.WriteLine("relative heading:");
                        file.WriteLine(GameplayCamera.RelativeHeading.ToString());
                        file.WriteLine("relative pitch:");
                        file.WriteLine(GameplayCamera.RelativePitch.ToString());
                        file.WriteLine("fov:");
                        file.WriteLine(GameplayCamera.FieldOfView.ToString());
                    }
                }
            }

            //if (k.KeyCode == Keys.T) // temp modification
            //{
            //    World.Weather = Weather.Raining;
            //    /* set it between 0 = stop, 1 = heavy rain. set it too high will lead to sloppy ground */
            //    Function.Call(GTA.Native.Hash._SET_RAIN_FX_INTENSITY, 0.5f);
            //    var test = Function.Call<float>(GTA.Native.Hash.GET_RAIN_LEVEL);
            //    UI.Notify("" + test);
            //    World.CurrentDayTime = new TimeSpan(12, 0, 0);
            //    //Script.Wait(5000);
            //}

            if (k.KeyCode == Keys.N)
            {
                /*
                //var color = VisionNative.GetColorBuffer();
                
                List<byte[]> colors = new List<byte[]>();
                Game.Pause(true);
                Script.Wait(1);
                var depth = VisionNative.GetDepthBuffer();
                var stencil = VisionNative.GetStencilBuffer();
                foreach (var wea in wantedWeather) {
                    World.TransitionToWeather(wea, 0.0f);
                    Script.Wait(1);
                    colors.Add(VisionNative.GetColorBuffer());
                }
                Game.Pause(false);
                if (depth != null)
                {
                    var res = Game.ScreenResolution;
                    var t = Tiff.Open(Path.Combine(dataPath, "test.tiff"), "w");
                    ImageUtils.WriteToTiff(t, res.Width, res.Height, colors, depth, stencil);
                    t.Close();
                    UI.Notify(GameplayCamera.FieldOfView.ToString());
                }
                else
                {
                    UI.Notify("No Depth Data quite yet");
                }
                //UI.Notify((connection != null && connection.Connected).ToString());
                */
                //var color = VisionNative.GetColorBuffer();

                var dateTimeFormat = @"yyyy-MM-dd_HH-mm-ss";

                for (int i = 0; i < 20000; i++)
                {
                    List<byte[]> colors = new List<byte[]>();
                    Game.Pause(true);

                    colors.Add(VisionNative.GetColorBuffer());
                    var depth = VisionNative.GetDepthBuffer();
                    var stencil = VisionNative.GetStencilBuffer();
                    Script.Wait(1);
                    //foreach (var wea in wantedWeather)
                    //{
                    //    World.TransitionToWeather(wea, 0.0f);
                    //    Script.Wait(1);
                    //    colors.Add(VisionNative.GetColorBuffer());
                    //}

                    //Game.Pause(false);

                    var data = GTAData.DumpData(Game.GameTime + ".dat", new List<Weather>(wantedWeather));
                    Script.Wait(0);

                    if (depth != null)
                    {
                        var res = Game.ScreenResolution;
                        var fileName = Path.Combine(dataPath, "gtav_" + i.ToString());
                        ImageUtils.WriteToTiff(fileName, res.Width, res.Height, colors, depth, stencil);
                        //UI.Notify(GameplayCamera.FieldOfView.ToString());
                        //UI.Notify((connection != null && connection.Connected).ToString());
                    }
                    else
                    {
                        UI.Notify("No Depth Data aquired yet");
                    }

                    path = @"D:\Devel\GTAVisionExport\managed\Data\ObjectList_" + FileCounter + ".json";
                    //This text is added only once to the file.
                    if (!File.Exists(path))
                    {
                        // Create a file to write to.
                        using (StreamWriter file = File.CreateText(path))
                        {
                            //file.WriteLine("File number " + FileCounter.ToString());
                            //file.WriteLine("All coordinates are in game space!");
                            //file.WriteLine("");
                        }
                    }

                    long length = new System.IO.FileInfo(path).Length;
          
                    if (length / 1024 / 1024 > 10) //if json file bigger than 10 MB
                    {
                        FileCounter++;
                        path = @"D:\Devel\GTAVisionExport\managed\Data\ObjectList_" + FileCounter + ".json";

                        if (!File.Exists(path))
                        {
                            // Create a file to write to.
                            using (StreamWriter file = File.CreateText(path))
                            {
                                //file.WriteLine("File number " + FileCounter.ToString());
                                //file.WriteLine("All coordinates are in game space!");
                                //file.WriteLine("");
                            }
                        }
                    }

                    CollectedData cd = new CollectedData
                    {
                        Image = "gtav_" + i.ToString() + ".tiff",
                        ImageWidth = data.ImageWidth,
                        ImageHeight = data.ImageHeight,
                        UIwidth = data.UIWidth,
                        UIheight = data.UIHeight,
                        RealTime = data.Timestamp,
                        GameTime = data.LocalTime,
                        GameTime2 = data.GameTime,
                        Campos = data.CamPos,
                        Camrot = data.CamRot,
                        Camdir = data.CamDirection,
                        PMatrix = data.ProjectionMatrix,
                        VMatrix = data.ViewMatrix,
                        Gamerpos = data.GamerPos,
                        Carpos = data.CarPos,
                        Carrot = data.CarRot,
                        CamHash = data.CamHash,
                        CamFOV = data.CamFOV,
                        CamNearClip = data.CamNearClip,
                        CamFarClip = data.CamFarClip
                    };

                    foreach (var detection in data.Detections)
                    {
                        cd.Detections.Add(detection);
                    }

                    string jsonData = JsonConvert.SerializeObject(cd);
                    File.AppendAllText(path, jsonData);
                    File.AppendAllText(path, "\n");

                    Game.Pause(false);
                    Script.Wait(0);
                }
            }

            //if (k.KeyCode == Keys.I)
            //{
            //    var info = new GTAVisionUtils.InstanceData();
            //    UI.Notify(info.type);
            //    UI.Notify(info.publichostname);
            //}
        }
    }
}