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
        public float FrameTime { get; set; }
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

    public class VisionExport : Script
    {
        private readonly Weather[] wantedWeather = new Weather[] { Weather.Clear, Weather.ExtraSunny, Weather.Overcast, Weather.Foggy, Weather.Clouds };
        private Player player;
        

        private UTF8Encoding encoding = new UTF8Encoding(false);
        private KeyHandling kh = new KeyHandling();
        private speedAndTime lowSpeedTime = new speedAndTime();

        // For future use
        private StereoCamera cams;

        // For the object lists
        private string path;
        private int FileCounter = 1;

        ImageUtils TiffSaver = new ImageUtils();
        public bool runActive = false;
        public int runCnt = 0;

        public VisionExport()
        {
            if (!Directory.Exists(SettingsReader.DATA_FOLDER)) Directory.CreateDirectory(SettingsReader.DATA_FOLDER);
            player = Game.Player;

            var parser = new FileIniDataParser();
            var data = new IniParser.Model.IniData();

            this.Tick += new EventHandler(this.OnTick);
            this.KeyDown += OnKeyDown;
            Interval = 0;

            // Activate Trainer
            player.IsInvincible = true;
            player.WantedLevel = 0;
            player.IgnoredByEveryone = true;
            player.IgnoredByPolice = true;
            player.SetSwimSpeedMultThisFrame(1.49f);
            player.SetRunSpeedMultThisFrame(1.49f);
            player.SetSuperJumpThisFrame();
            //PED_FLAG_CAN_FLY_THRU_WINDSCREEN
            HashFunctions.SetPedConfigFlag(Game.Player.Character, 32, false);
            //_PED_SWITCHING_WEAPON
            HashFunctions.SetPedConfigFlag(Game.Player.Character, 331, false);
            HashFunctions.SpecialAbilityFillMeter(player, true);
            HashFunctions.SetPlayerNoiseMultiplier(player, false);
            HashFunctions.SetCreateRandomCops(false);
            HashFunctions.SetRandomBoats(false);
            HashFunctions.SetRandomTrains(false);
            HashFunctions.SetGarbageTrucks(false);
            World.CurrentDayTime = new TimeSpan(13, 0, 0);
            Game.PauseClock(true);
            World.WeatherTransition = 0;
        }

        public void OnTick(object o, EventArgs e)
        {
            while (runActive && runCnt <= SettingsReader.runLoop)
            {
                Game.Pause(true);
                runDataCollection(runCnt);
                runCnt += 1;
                Game.Pause(false);
                Script.Wait(0);
            }
        }

        public void runDataCollection(int i)
        {
            var dateTimeFormat = @"yyyy-MM-dd_HH-mm-ss";
            int crossID = 0;

            Crossings.getCrossID(out crossID);
            string fname = "gtav_cid" + crossID.ToString() + "_c" + World.RenderingCamera.Handle + "_" + i.ToString();
            //List<byte[]> colors = new List<byte[]>();

            //colors.Add(VisionNative.GetColorBuffer());
            var colors = VisionNative.GetColorBuffer();
            var depth = VisionNative.GetDepthBuffer();
            var stencil = VisionNative.GetStencilBuffer();
            Script.Wait(1);
            //foreach (var wea in wantedWeather)
            //{
            //    World.TransitionToWeather(wea, 0.0f);
            //    Script.Wait(1);
            //    colors.Add(VisionNative.GetColorBuffer());
            //}

            if (depth != null && stencil!= null && colors != null)
            {
                var data = GTAData.DumpData(new List<Weather>(wantedWeather));
                Script.Wait(0);

                var res = Game.ScreenResolution;
                var fileName = Path.Combine(SettingsReader.DATA_FOLDER, fname);
                //UI.ShowSubtitle(fileName);
                TiffSaver.WriteToTiff(fileName, res.Width, res.Height, colors, depth, stencil);
                //Script.Wait(0);

                path = SettingsReader.DATA_FOLDER + "ObjectList_" + FileCounter + ".json";
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

                if (length / 1024 / 1024 > 20) //if json file bigger than 20 MB
                {
                    FileCounter++;
                    path = SettingsReader.DATA_FOLDER + "ObjectList_" + FileCounter + ".json";

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
                    Image = fname + ".tiff",
                    ImageWidth = data.ImageWidth,
                    ImageHeight = data.ImageHeight,
                    UIwidth = data.UIWidth,
                    UIheight = data.UIHeight,
                    RealTime = data.Timestamp,
                    GameTime = data.LocalTime,
                    GameTime2 = data.GameTime,
                    FrameTime = data.FrameTime,
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
            }
            else
            {
                UI.Notify("No Depth Data aquired yet");
            }
            
            //Script.Wait(0);
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
            Model mod = new Model(GTA.Native.VehicleHash.Adder);
            var vehicle = GTA.World.CreateVehicle(mod, player.Character.Position);
            vehicle.CanTiresBurst = false;
            vehicle.CanBeVisiblyDamaged = false;
            vehicle.CanWheelsBreak = false;
            vehicle.IsInvincible = true;

            vehicle.PrimaryColor = VehicleColor.MetallicBlack;
            vehicle.SecondaryColor = VehicleColor.MetallicOrange;
            vehicle.PlaceOnGround();
            vehicle.NumberPlate = " Cuky ";

            player.Character.SetIntoVehicle(vehicle, VehicleSeat.Driver);
        }

        public void ToggleNavigation()
        {
            //YOLO
            MethodInfo inf = kh.GetType().GetMethod("AtToggleAutopilot", BindingFlags.NonPublic | BindingFlags.Instance);
            inf.Invoke(kh, new object[] { new KeyEventArgs(Keys.J) });
        }

        public void ReloadGame()
        {
            // no need to release the autodrive here
            // delete all surrounding vehicles & the driver's car
            HashFunctions.ClearAreaOfVehicle(player.Character.Position, 1000f, false, false, false, false);
            player.LastVehicle.Delete();
            // teleport to the spawning position, defined in GameUtils.cs, subject to changes
            player.Character.Position = GTAConst.StartPos;
            HashFunctions.ClearAreaOfVehicle(player.Character.Position, 100f, false, false, false, false);
            // start a new run
            EnterVehicle();
            //Script.Wait(2000);
            ToggleNavigation();

            lowSpeedTime.clearTime();
        }

        public void OnKeyDown(object o, KeyEventArgs k)
        {
            if (k.KeyCode == Keys.Y) // temp modification
            {
                ReloadGame();
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
                if (runActive)
                {
                    runActive = false;
                }
                else
                {
                    runActive = true;
                }
            }
        }
    }
}