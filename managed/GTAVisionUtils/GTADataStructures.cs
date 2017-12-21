using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Policy;
using GTA;
using GTA.Math;
using System.Globalization;
using GTA.Native;
using Npgsql;
using SharpDX;
using SharpDX.Mathematics;
using NativeUI;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;
using Vector2 = GTA.Math.Vector2;
using Vector3 = GTA.Math.Vector3;
using Point = System.Drawing.Point;
using System.IO;

namespace GTAVisionUtils
{
    public class GTARun
    {
        public Guid guid { get; set; }
        public string archiveKey { get; set; }
    }

    public class GTABoundingBox2
    {
        public GTAVector2 Min { get; set; }
        public GTAVector2 Max { get; set; }
        public float Area {
            get {
                return (Max.X - Min.X) * (Max.Y - Min.Y);
            }
        }
    }

    public class GTAVehicle
    {
        public GTAVector Pos { get; set; }
        public GTABoundingBox2 BBox { get; set; }

        public GTAVehicle(Vehicle v)
        {
            Pos = new GTAVector(v.Position);
            BBox = GTAData.ComputeBoundingBox(v, v.Position);
        }
    }

    public class GTAPed {
        public GTAVector Pos { get; set; }
        public GTABoundingBox2 BBox { get; set; }
        public GTAPed(Ped p)
        {
            Pos = new GTAVector(p.Position);
            BBox = GTAData.ComputeBoundingBox(p, p.Position);
        }
    }

    public enum DetectionType
    {
        background,
        person,
        car,
        bicycle
    }
    public enum DetectionClass {
        Unknown = -1,
        Compacts = 0,
        Sedans = 1,
        SUVs = 2,
        Coupes = 3,
        Muscle = 4,
        SportsClassics = 5,
        Sports = 6,
        Super = 7,
        Motorcycles = 8,
        OffRoad = 9,
        Industrial = 10,
        Utility = 11,
        Vans = 12,
        Cycles = 13,
        Boats = 14,
        Helicopters = 15,
        Planes = 16,
        Service = 17,
        Emergency = 18,
        Military = 19,
        Commercial = 20,
        Trains = 21
    }

    public class GTADetection
    {
        public string Type { get; set; }
        public string cls { get; set; }
        public GTAVector Pos { get; set; }
        public GTAVector Rot { get; set; }
        public GTAVector Dim { get; set; }
        public float Distance { get; set; }
        public float Speed { get; set; }
        public float Wheel { get; set; }
        public int HashCode { get; set; }
        //public int ImageWidth { get; set; }
        //public int ImageHeight { get; set; }
        public bool Visibility { get; set; }
        public string NumberPlate { get; set; }
        // public GTABoundingBox2 BBox { get; set; }
        // public BoundingBox BBox3D { get; set; }
        public int Handle { get; set; }
        public GTAVector2 FUR;
        public GTAVector2 FUL;
        public GTAVector2 BUL;
        public GTAVector2 BUR;
        public GTAVector2 FLR;
        public GTAVector2 FLL;
        public GTAVector2 BLL;
        public GTAVector2 BLR;
        public GTAVector FURGame;
        public GTAVector FULGame;
        public GTAVector BULGame;
        public GTAVector BURGame;
        public GTAVector FLRGame;
        public GTAVector FLLGame;
        public GTAVector BLLGame;
        public GTAVector BLRGame;
        public GTAVector RightVector;
        public GTAVector ForwardVector;
        public GTAVector UpVector;
        public float GroundHeight1 { get; set; }
        public float GroundHeight2 { get; set; }
        public float GroundHeight3 { get; set; }
        public float GroundHeight4 { get; set; }
        public float GroundHeight5 { get; set; }
        public float GroundHeight6 { get; set; }
        public float GroundHeight7 { get; set; }
        public float GroundHeight8 { get; set; }

        public GTADetection(Entity e, DetectionType type, int ImgW, int ImgH, GTAVector CamPos)
        {
            Type = type.ToString();
            Pos = new GTAVector(e.Position);
            Distance = Game.Player.Character.Position.DistanceTo(new Vector3(Pos.X, Pos.Y, Pos.Z));
            
            // BBox = GTAData.ComputeBoundingBox(e, e.Position);

            Handle = e.Handle;
            HashCode = e.GetHashCode();

            Rot = new GTAVector(e.Rotation);
            cls = "Unknown";
            Vector3 gmin;
            Vector3 gmax;
            e.Model.GetDimensions(out gmin, out gmax);
            Dim = new GTAVector(gmax - gmin);

            //BoundingBox BBox3DGame = new SharpDX.BoundingBox((SharpDX.Vector3)new GTAVector(gmin), (SharpDX.Vector3)new GTAVector(gmax));
            //FURGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[0].X, BBox3DGame.GetCorners()[0].Y, BBox3DGame.GetCorners()[0].Z)));
            //FULGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[1].X, BBox3DGame.GetCorners()[1].Y, BBox3DGame.GetCorners()[1].Z)));
            //BULGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[2].X, BBox3DGame.GetCorners()[2].Y, BBox3DGame.GetCorners()[2].Z)));
            //BURGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[3].X, BBox3DGame.GetCorners()[3].Y, BBox3DGame.GetCorners()[3].Z)));
            //FLRGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[4].X, BBox3DGame.GetCorners()[4].Y, BBox3DGame.GetCorners()[4].Z)));
            //FLLGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[5].X, BBox3DGame.GetCorners()[5].Y, BBox3DGame.GetCorners()[5].Z)));
            //BLLGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[6].X, BBox3DGame.GetCorners()[6].Y, BBox3DGame.GetCorners()[6].Z)));
            //BLRGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[7].X, BBox3DGame.GetCorners()[7].Y, BBox3DGame.GetCorners()[7].Z)));

            Vector3 RightV = e.RightVector;
            Vector3 ForwardV = e.ForwardVector;
            Vector3 UpV = e.UpVector;

            RightVector = new GTAVector(RightV);
            ForwardVector = new GTAVector(ForwardV);
            UpVector = new GTAVector(UpV);

            FURGame = new GTAVector(Pos.X + (Dim.X / 2) * RightVector.X + (Dim.Y / 2) * ForwardVector.X + (Dim.Z / 2) * UpVector.X, Pos.Y + (Dim.X / 2) * RightVector.Y + (Dim.Y / 2) * ForwardVector.Y + (Dim.Z / 2) * UpVector.Y, Pos.Z + (Dim.X / 2) * RightVector.Z + (Dim.Y / 2) * ForwardVector.Z + (Dim.Z / 2) * UpVector.Z);
            //FURGame.X = Pos.X + (Dim.X / 2) * RightVector.X + (Dim.Y / 2) * ForwardVector.X + (Dim.Z / 2) * UpVector.X;
            //FURGame.Y = Pos.Y + (Dim.X / 2) * RightVector.Y + (Dim.Y / 2) * ForwardVector.Y + (Dim.Z / 2) * UpVector.Y;
            //FURGame.Z = Pos.Z + (Dim.X / 2) * RightVector.Z + (Dim.Y / 2) * ForwardVector.Z + (Dim.Z / 2) * UpVector.Z;

            BLLGame = new GTAVector(Pos.X - (Dim.X / 2) * RightVector.X - (Dim.Y / 2) * ForwardVector.X - (Dim.Z / 2) * UpVector.X, Pos.Y - (Dim.X / 2) * RightVector.Y - (Dim.Y / 2) * ForwardVector.Y - (Dim.Z / 2) * UpVector.Y, Pos.Z - (Dim.X / 2) * RightVector.Z - (Dim.Y / 2) * ForwardVector.Z - (Dim.Z / 2) * UpVector.Z);
            //BLLGame.X = Pos.X - (Dim.X / 2) * RightVector.X - (Dim.Y / 2) * ForwardVector.X - (Dim.Z / 2) * UpVector.X;
            //BLLGame.Y = Pos.Y - (Dim.X / 2) * RightVector.Y - (Dim.Y / 2) * ForwardVector.Y - (Dim.Z / 2) * UpVector.Y;
            //BLLGame.Z = Pos.Z - (Dim.X / 2) * RightVector.Z - (Dim.Y / 2) * ForwardVector.Z - (Dim.Z / 2) * UpVector.Z;

            GTAVector dummy = new GTAVector(Dim.X * RightV);
            FULGame = new GTAVector(FURGame.X - dummy.X, FURGame.Y - dummy.Y, FURGame.Z - dummy.Z);

            dummy = new GTAVector(Dim.Y * Vector3.Cross(UpV, RightV));
            BURGame = new GTAVector(FURGame.X - dummy.X, FURGame.Y - dummy.Y, FURGame.Z - dummy.Z);

            dummy = new GTAVector(Dim.X * RightV);
            BULGame = new GTAVector(BURGame.X - dummy.X, BURGame.Y - dummy.Y, BURGame.Z - dummy.Z);

            dummy = new GTAVector(Dim.X * RightV);
            BLRGame = new GTAVector(BLLGame.X + dummy.X, BLLGame.Y + dummy.Y, BLLGame.Z + dummy.Z);

            dummy = new GTAVector(Dim.Y * Vector3.Cross(UpV, RightV));
            FLLGame = new GTAVector(BLLGame.X + dummy.X, BLLGame.Y + dummy.Y, BLLGame.Z + dummy.Z);

            dummy = new GTAVector(Dim.X * RightV);
            FLRGame = new GTAVector(FLLGame.X + dummy.X, FLLGame.Y + dummy.Y, FLLGame.Z + dummy.Z);

            GroundHeight1 = World.GetGroundHeight(new Vector3(FURGame.X, FURGame.Y, FURGame.Z));
            GroundHeight2 = World.GetGroundHeight(new Vector3(FULGame.X, FULGame.Y, FULGame.Z));
            GroundHeight3 = World.GetGroundHeight(new Vector3(BULGame.X, BULGame.Y, BULGame.Z));
            GroundHeight4 = World.GetGroundHeight(new Vector3(BURGame.X, BURGame.Y, BURGame.Z));
            GroundHeight5 = World.GetGroundHeight(new Vector3(FLRGame.X, FLRGame.Y, FLRGame.Z));
            GroundHeight6 = World.GetGroundHeight(new Vector3(FLLGame.X, FLLGame.Y, FLLGame.Z));
            GroundHeight7 = World.GetGroundHeight(new Vector3(BLLGame.X, BLLGame.Y, BLLGame.Z));
            GroundHeight8 = World.GetGroundHeight(new Vector3(BLRGame.X, BLRGame.Y, BLRGame.Z));

            FLRGame.Z = FLRGame.Z - (FLRGame.Z - GroundHeight5);
            FLLGame.Z = FLLGame.Z - (FLLGame.Z - GroundHeight6);
            BLLGame.Z = BLLGame.Z - (BLLGame.Z - GroundHeight7);
            BLRGame.Z = BLRGame.Z - (BLRGame.Z - GroundHeight8);

            FUR = GTAData.get2Dfrom3D(new Vector3(FURGame.X, FURGame.Y, FURGame.Z), ImgW, ImgH);
            FUL = GTAData.get2Dfrom3D(new Vector3(FULGame.X, FULGame.Y, FULGame.Z), ImgW, ImgH);
            BUL = GTAData.get2Dfrom3D(new Vector3(BULGame.X, BULGame.Y, BULGame.Z), ImgW, ImgH);
            BUR = GTAData.get2Dfrom3D(new Vector3(BURGame.X, BURGame.Y, BURGame.Z), ImgW, ImgH);
            FLR = GTAData.get2Dfrom3D(new Vector3(FLRGame.X, FLRGame.Y, FLRGame.Z), ImgW, ImgH);
            FLL = GTAData.get2Dfrom3D(new Vector3(FLLGame.X, FLLGame.Y, FLLGame.Z), ImgW, ImgH);
            BLL = GTAData.get2Dfrom3D(new Vector3(BLLGame.X, BLLGame.Y, BLLGame.Z), ImgW, ImgH);
            BLR = GTAData.get2Dfrom3D(new Vector3(BLRGame.X, BLRGame.Y, BLRGame.Z), ImgW, ImgH);

            Visibility = GTAData.visibleOnScreen(new GTAVector[] { FURGame, BLLGame, FULGame, BURGame, BULGame, BLRGame, FLLGame, FLRGame }, e, CamPos);
            //Visibility = GTAData.CheckVisible(e);

            //World.DrawMarker(MarkerType.DebugSphere, FURGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Green);
            //World.DrawMarker(MarkerType.DebugSphere, FULGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Yellow);
            //World.DrawMarker(MarkerType.DebugSphere, BULGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Pink);
            //World.DrawMarker(MarkerType.DebugSphere, BURGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Red);
            //World.DrawMarker(MarkerType.DebugSphere, FLRGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.White);
            //World.DrawMarker(MarkerType.DebugSphere, FLLGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Orange);
            //World.DrawMarker(MarkerType.DebugSphere, BLLGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Black);
            //World.DrawMarker(MarkerType.DebugSphere, BLRGame, new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(.1f, .1f, .1f), System.Drawing.Color.Blue);
        }

        public GTADetection(Ped p, int ImgW, int ImgH, GTAVector CamPos) : this(p, DetectionType.person, ImgW, ImgH, CamPos)
        {
        }

        public GTADetection(Vehicle v, int ImgW, int ImgH, GTAVector CamPos) : this(v, DetectionType.car, ImgW, ImgH, CamPos)
        {
            cls = v.ClassType.ToString();
            Speed = (float)(v.Speed * 3.6);
            Wheel = v.SteeringAngle;
            NumberPlate = v.NumberPlate;
        }
    }

    public class GTAVector
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public GTAVector(Vector3 v)
        {
            X = v.X;
            Y = v.Y;
            Z = v.Z;
        }

        public GTAVector(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static explicit operator SharpDX.Vector3(GTAVector i)
        {
            return new SharpDX.Vector3(i.X, i.Y, i.Z);
        }
    }

    public class GTAVector2
    {
        public float X { get; set; }
        public float Y { get; set; }

        public GTAVector2(float x, float y)
        {
            X = x;
            Y = y;
        }
        public GTAVector2()
        {
            X = 0f;
            Y = 0f;
        }
        public GTAVector2(Vector2 v)
        {
            X = v.X;
            Y = v.Y;
        }
    }

    public class GTAData
    {
        public int Version { get; set; }
        public string ImageName { get; set; }
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public int UIHeight { get; set; }
        public int UIWidth { get; set; }
        public int CamHash { get; set; }
        public DateTime Timestamp { get; set; }
        public TimeSpan LocalTime { get; set; }
        public int GameTime { get; set; }
        public Weather CurrentWeather { get; set; }
        public List<Weather> CapturedWeathers;
        public GTAVector CamPos { get; set; }
        public GTAVector CamRot { get; set; }
        public GTAVector GamerPos { get; set; }
        public GTAVector CamDirection { get; set; }
        public GTAVector CarPos { get; set; }
        public GTAVector CarRot { get; set; }
        //mathnet's matrices are in heap storage, which is super annoying, 
        //but we want to use double matrices to avoid numerical issues as we
        //decompose the MVP matrix into seperate M,V and P matrices
        public DenseMatrix ViewMatrix { get; set; }
        public DenseMatrix ProjectionMatrix { get; set; }
        public double CamFOV { get; set; }
        public double CamNearClip { get; set; }
        public double CamFarClip { get; set; }
        public List<GTADetection> Detections { get; set; }

        public static SharpDX.Vector3 CvtVec(GTA.Math.Vector3 inp) {
            return (SharpDX.Vector3)new GTAVector(inp);

        }

        public static Vector2 scalePoints(Vector2 p, int ImageWidth, int ImageHeight)
        {
            // Arthur
            return new Vector2((int)(ImageWidth / (1.0 * UI.WIDTH) * p.X), (int)(ImageHeight / (1.0 * UI.HEIGHT) * p.Y));
            // Mem lookup: http://gtaforums.com/topic/842182-world-to-screen-lag/; 3DBB passt, 2DBB ist falsch!!!
            // return new Vector2((int)(UI.WIDTH * 0.5f + p.X * UI.WIDTH * 0.5f), (int)(UI.HEIGHT * 0.5f + p.Y * UI.HEIGHT * 0.5f));
            // Test
            //return new Vector2((int)(p.X * UI.WIDTH), (int)(p.Y * UI.HEIGHT));
        }

        public static GTAVector2 get2Dfrom3D(Vector3 a, int ImageWidth, int ImageHeight)
        {
            // http://orfe.princeton.edu/~alaink/SmartDrivingCars/Visitors/FilipowiczVideoGamesforAutonomousDriving.pdf
            // camera rotation 
            Vector3 theta = Vector3.Zero;
            theta = (float)(System.Math.PI / 180f) * World.RenderingCamera.Rotation;

            // camera direction, at 0 rotation the camera looks down the postive Y axis --> WorldNorth schaut somit immer in Cam-Richtung
            Vector3 camDir = rotate(Vector3.WorldNorth, theta);

            // camera position  == eigentlich Bildebene rotiert in Blickrichtung, minimal vor der Cam zu sehen
            Vector3 c = Vector3.Zero;
            c = World.RenderingCamera.Position + World.RenderingCamera.NearClip * camDir;

            // viewer position == Cam-Pos rotiert in Blickrichtung; minimal hinter c!
            Vector3 e = -World.RenderingCamera.NearClip * camDir;

            float viewWindowHeight = 2 * World.RenderingCamera.NearClip * (float)System.Math.Tan((World.RenderingCamera.FieldOfView / 2f) * (System.Math.PI / 180f));
            float viewWindowWidth = (ImageWidth / ((float)ImageHeight)) * viewWindowHeight;

            Vector3 camUp = rotate(Vector3.WorldUp, theta);
            Vector3 camEast = rotate(Vector3.WorldEast, theta);

            // Distanz zwischen Punkt und Bildebene
            Vector3 del = a - c;

            Vector3 viewerDist = del - e;
            //Vector3 viewerDistNorm = viewerDist * (1 / viewerDist.Length());
            Vector3 viewerDistNorm = Vector3.Normalize(viewerDist);
            float dot = Vector3.Dot(camDir, viewerDistNorm);
            float ang = (float)System.Math.Acos((double)dot);
            // Senkrechter Abstand zur Bildebene
            float viewPlaneDist = World.RenderingCamera.NearClip / (float)System.Math.Cos((double)ang);
            // Punkt auf der Bildebene
            Vector3 viewPlanePoint = viewPlaneDist * viewerDistNorm + e;

            // move origin to upper left 
            Vector3 newOrigin = c + (viewWindowHeight / 2f) * camUp - (viewWindowWidth / 2f) * camEast;
            viewPlanePoint = (viewPlanePoint + c) - newOrigin;

            float viewPlaneX = Vector3.Dot(viewPlanePoint, camEast) / Vector3.Dot(camEast, camEast);
            float viewPlaneZ = Vector3.Dot(viewPlanePoint, camUp) / Vector3.Dot(camUp, camUp);

            float screenX = viewPlaneX / viewWindowWidth * UI.WIDTH;
            float screenY = -viewPlaneZ / viewWindowHeight * UI.HEIGHT;

            Vector2 screenScale = scalePoints(new Vector2(screenX, screenY), ImageWidth, ImageHeight);

            /*
            string path = @"D:\Devel\GTAVisionExport\managed\Data\transformation.txt";
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
            // Create a file to write to.
            using (StreamWriter file = File.AppendText(path))
                {
                    file.WriteLine("Vector: " + a.ToString());
                    file.WriteLine("theta: " + theta.ToString());
                    file.WriteLine("camDir: " + camDir.ToString());
                    file.WriteLine("c: " + c.ToString());
                    file.WriteLine("e: " + e.ToString());
                    file.WriteLine("viewWindowHeight: " + viewWindowHeight.ToString());
                    file.WriteLine("viewWindowWidth: " + viewWindowWidth.ToString());
                    file.WriteLine("camUp: " + camUp.ToString());
                    file.WriteLine("camEast: " + camEast.ToString());
                    file.WriteLine("del: " + del.ToString());
                    file.WriteLine("viewerDist: " + viewerDist.ToString());
                    file.WriteLine("viewerDistNorm: " + viewerDistNorm.ToString());
                    file.WriteLine("dot: " + dot.ToString());
                    file.WriteLine("ang: " + ang.ToString());
                    file.WriteLine("viewPlaneDist: " + viewPlaneDist.ToString());
                    file.WriteLine("viewPlanePoint: " + viewPlanePoint.ToString());
                    file.WriteLine("newOrigin: " + newOrigin.ToString());
                    file.WriteLine("viewPlanePoint: " + viewPlanePoint.ToString());
                    file.WriteLine("viewPlaneX: " + viewPlaneX.ToString());
                    file.WriteLine("viewPlaneZ: " + viewPlaneZ.ToString());
                    file.WriteLine("screenX: " + screenX.ToString());
                    file.WriteLine("screenY: " + screenY.ToString());
                    file.WriteLine("Xscale: " + screenScale.X.ToString());
                    file.WriteLine("Yscale: " + screenScale.Y.ToString());

                }
                */

            //return new Vector2((int)screenX, (int)screenY);
            return new GTAVector2(screenScale.X, screenScale.Y);
        }

        public static Vector3 rotate(Vector3 a, Vector3 theta)
        {
            Vector3 d = new Vector3();

            // Rotation order: Z Y X
            d.X = (float)System.Math.Cos((double)theta.Z) * ((float)System.Math.Cos((double)theta.Y) * a.X + (float)System.Math.Sin((double)theta.Y) * ((float)System.Math.Sin((double)theta.X) * a.Y + (float)System.Math.Cos((double)theta.X) * a.Z)) - (float)System.Math.Sin((double)theta.Z) * ((float)System.Math.Cos((double)theta.X) * a.Y - (float)System.Math.Sin((double)theta.X) * a.Z);
            d.Y = (float)System.Math.Sin((double)theta.Z) * ((float)System.Math.Cos((double)theta.Y) * a.X + (float)System.Math.Sin((double)theta.Y) * ((float)System.Math.Sin((double)theta.X) * a.Y + (float)System.Math.Cos((double)theta.X) * a.Z)) + (float)System.Math.Cos((double)theta.Z) * ((float)System.Math.Cos((double)theta.X) * a.Y - (float)System.Math.Sin((double)theta.X) * a.Z);
            d.Z = -(float)System.Math.Sin((double)theta.Y) * a.X + (float)System.Math.Cos((double)theta.Y) * ((float)System.Math.Sin((double)theta.X) * a.Y + (float)System.Math.Cos((double)theta.X) * a.Z);

            return d;
        }

        public static GTABoundingBox2 ComputeBoundingBox(Entity e, Vector3 offset, float scale = 0.5f)
        {
            
            var m = e.Model;
            var rv = new GTABoundingBox2
            {
                Min = new GTAVector2(float.PositiveInfinity, float.PositiveInfinity),
                Max = new GTAVector2(float.NegativeInfinity, float.NegativeInfinity)
            };
            Vector3 gmin;
            Vector3 gmax;
            m.GetDimensions(out gmin, out gmax);
            var bbox = new SharpDX.BoundingBox((SharpDX.Vector3)new GTAVector(gmin), (SharpDX.Vector3)new GTAVector(gmax));
            //Console.WriteLine(bbox.GetCorners()[0]);
            
            //for (int i = 0; i < bbox.GetCorners().Length; ++i) {
            //    for (int j = 0; j < bbox.GetCorners().Length; ++j) {
            //        if (j == i) continue;
            //        var c1 = bbox.GetCorners()[i];
            //        var c2 = bbox.GetCorners()[j];
            //        HashFunctions.Draw3DLine(e.GetOffsetInWorldCoords(new Vector3(c1.X, c1.Y, c1.Z)), e.GetOffsetInWorldCoords(new Vector3(c2.X, c2.Y, c2.Z)), 0,0);
            //    }
            //}
            
            /*
            for (int i = 0; i < bbox.GetCorners().Length; ++i)
            {
                var corner = bbox.GetCorners()[i];
                var cornerinworld = e.GetOffsetInWorldCoords(new Vector3(corner.X, corner.Y, corner.Z));


            }*/
            //UI.Notify(e.HeightAboveGround.ToString());
            var sp = HashFunctions.Convert3dTo2d(e.GetOffsetInWorldCoords(e.Position));
            foreach (var corner in bbox.GetCorners()) {
                var c = new Vector3(corner.X, corner.Y, corner.Z);

                c = e.GetOffsetInWorldCoords(c);
                var s = HashFunctions.Convert3dTo2d(c);
                if (s.X == -1f || s.Y == -1f)
                {
                    rv.Min.X = float.PositiveInfinity;
                    rv.Max.X = float.NegativeInfinity;
                    rv.Min.Y = float.PositiveInfinity;
                    rv.Max.Y = float.NegativeInfinity;
                    return rv;
                }
                /*
                if(s.X == -1) {
                    if (sp.X < 0.5) s.X = 0f;
                    if (sp.X >= 0.5) s.X = 1f;
                }
                if(s.Y == -1) {
                    if (sp.Y < 0.5) s.Y = 0f;
                    if (sp.Y >= 0.5) s.Y = 1f;
                }
                */
                rv.Min.X = Math.Min(rv.Min.X, s.X);
                rv.Min.Y = Math.Min(rv.Min.Y, s.Y);
                rv.Max.X = Math.Max(rv.Max.X, s.X);
                rv.Max.Y = Math.Max(rv.Max.Y, s.Y);
            }
            
            //int x = (int)(rv.Min.X * 1600);
            //int y = (int)(rv.Min.Y * 1024);
            //int x2 = (int)(rv.Max.X);
            //int y2 = (int)(rv.Max.Y * 1024);
            //float w = rv.Max.X - rv.Min.X;
            //float h = rv.Max.Y - rv.Min.Y;
            //HashFunctions.DrawRect(rv.Min.X + w/2, rv.Min.Y + h/2, rv.Max.X - rv.Min.X, rv.Max.Y - rv.Min.Y, 255, 255, 255, 100);

            //new UIRectangle(new Point((int)(rv.Min.X * 1920), (int)(rv.Min.Y * 1080)), rv.)
            return rv;
        }

        public static bool visibleOnScreen(GTAVector[] vertices, Entity e, GTAVector CamPos)
        {
            //sw.Restart();
            bool isOnScreen = false;
            int cnt = 0;
            Vector3 f = new Vector3(CamPos.X, CamPos.Y, CamPos.Z);
            //UI.ShowSubtitle(World.RenderingCamera.Position.ToString() + "\n" + Game.Player.LastVehicle.Position.ToString(), 1000);
            foreach (GTAVector v in vertices)
            {
                
                Vector3 vec = new Vector3(v.X, v.Y, v.Z);
                //UI.Notify(UI.WorldToScreen(vec).ToString());
                if (UI.WorldToScreen(vec).X != 0 && UI.WorldToScreen(vec).Y != 0)
                {
                    // is if point is visible on screen
                    //f = World.RenderingCamera.Position;

                    Vector3 h = World.Raycast(f, vec, IntersectOptions.Everything).HitCoords;

                    if ((h - f).Length() < (vec - f).Length())
                    {

                    }
                    else
                    {
                        cnt += 1;
                        //break;
                    }
                }
            }

            if(Vector3.Distance(f, e.Position) < 15f & cnt > 1)
            {
                return true;
            }
            else if (Vector3.Distance(f, e.Position) < 50f & cnt > 2)
            {
                return true;
            }
            else if (Vector3.Distance(f, e.Position) < 100f & cnt > 4)
            {
                return true;
            }
            else
            {
                return false;
            }

            //UI.ShowSubtitle("IsVisible: " + sw.Elapsed, 1000);
            //UI.ShowSubtitle(isOnScreen.ToString(), 500);
        }

        public static bool CheckVisible(Entity e) {
            //return true;
            //var p = Game.Player.LastVehicle;

            if (UI.WorldToScreen(e.Position).X != 0 && UI.WorldToScreen(e.Position).Y != 0)
            {
                //var ppos = GameplayCamera.Position;
                var ppos = World.RenderingCamera.Position;
                Vector3 h = World.Raycast(ppos, e.Position, IntersectOptions.Everything).HitCoords;
                var isLOS = Function.Call<bool>((GTA.Native.Hash)0x0267D00AF114F17A, Game.Player.Character, e);

                if ((h - ppos).Length() < (e.Position + (e.Model.GetDimensions().Y/2) * Vector3.Cross(e.UpVector, e.RightVector) - ppos).Length())
                {
                    return false;
                }
                else
                {
                    return true;
                }

                return isLOS;
                //var ppos = GameplayCamera.Position;

                //var res = World.Raycast(ppos, e.Position, IntersectOptions.Everything, Game.Player.Character.CurrentVehicle);
                //HashFunctions.Draw3DLine(ppos, e.Position);
                //UI.Notify("Camera: " + ppos.X + " Ent: " + e.Position.X);
                //World.DrawMarker(MarkerType.HorizontalCircleSkinny_Arrow, p.Position, (e.Position - p.Position).Normalized, Vector3.Zero, new Vector3(1, 1, 1), System.Drawing.Color.Red);
                //return res.HitEntity == e;
                //if (res.HitCoords == null) return false;
                //return e.IsInRangeOf(res.HitCoords, 10);
                //return res.HitEntity == e;
            }
            return false;
        }

        public static GTAData DumpData(string imageName, List<Weather> capturedWeathers)
        {
            var ret = new GTAData();
            ret.Version = 3;
            ret.ImageName = imageName;
            ret.CurrentWeather = World.Weather;
            ret.CapturedWeathers = capturedWeathers;
            
            ret.Timestamp = DateTime.UtcNow;
            ret.LocalTime = World.CurrentDayTime;
            ret.GameTime = Game.GameTime;
            ret.CamPos = new GTAVector(World.RenderingCamera.Position);
            ret.CamRot = new GTAVector(World.RenderingCamera.Rotation);
            ret.CamDirection = new GTAVector(World.RenderingCamera.Direction);
            ret.CamHash = World.RenderingCamera.GetHashCode();
            ret.CamFOV = World.RenderingCamera.FieldOfView;
            ret.ImageWidth = Game.ScreenResolution.Width;
            ret.ImageHeight = Game.ScreenResolution.Height;
            ret.UIWidth = UI.WIDTH;
            ret.UIHeight = UI.HEIGHT;
            ret.GamerPos = new GTAVector(Game.Player.Character.Position);
            ret.CamNearClip = World.RenderingCamera.NearClip;
            ret.CamFarClip = World.RenderingCamera.FarClip;

            if (Game.Player.Character.IsInVehicle())
            {
                ret.CarPos = new GTAVector(Game.Player.Character.CurrentVehicle.Position);
                ret.CarRot = new GTAVector(Game.Player.Character.CurrentVehicle.Rotation);
            }
            else
            {
                ret.CarPos = new GTAVector(Vector3.Zero);
                ret.CarRot = new GTAVector(Vector3.Zero);
            }
            
            var peds = World.GetNearbyPeds(Game.Player.Character, 150.0f);
            var cars = World.GetNearbyVehicles(Game.Player.Character, 150.0f);
            //var props = World.GetNearbyProps(Game.Player.Character.Position, 300.0f);
            
            var constants = VisionNative.GetConstants();
            if (!constants.HasValue) return null;
            var W = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4, constants.Value.world.ToArray()).ToDouble();
            var WV =
                MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4,
                    constants.Value.worldView.ToArray()).ToDouble();
            var WVP =
                MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4,
                    constants.Value.worldViewProjection.ToArray()).ToDouble();

            var V = WV*W.Inverse();
            var P = WVP*WV.Inverse();
            ret.ProjectionMatrix = P as DenseMatrix;
            ret.ViewMatrix = V as DenseMatrix;
            
            var pedList = from ped in peds
                where ped.IsHuman && ped.IsOnFoot && ped != Game.Player.Character
                          select new GTADetection(ped, ret.ImageWidth, ret.ImageHeight, ret.CamPos);
            var cycles = from ped in peds
                where ped.IsOnBike && ped != Game.Player.Character
                         select new GTADetection(ped, DetectionType.bicycle, ret.ImageWidth, ret.ImageHeight, ret.CamPos);
            
            var vehicleList = from car in cars
                              where car != Game.Player.Character.CurrentVehicle
                              select new GTADetection(car, ret.ImageWidth, ret.ImageHeight, ret.CamPos);
            ret.Detections = new List<GTADetection>();
            ret.Detections.AddRange(pedList);
            ret.Detections.AddRange(vehicleList);
            ret.Detections.AddRange(cycles);
            
            return ret;
        }
        
    }
}