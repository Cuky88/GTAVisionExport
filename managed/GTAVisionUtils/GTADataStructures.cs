using System;
using System.Collections.Generic;
using System.Linq;
using GTA;
using GTA.Native;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using Vector2 = GTA.Math.Vector2;
using Vector3 = GTA.Math.Vector3;
using SharpDX;
using System.IO;

namespace GTAVisionUtils
{
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
        public bool Visibility { get; set; }
        public int DistCat { get; set; }
        public string NumberPlate { get; set; }
        // public GTABoundingBox2 BBox { get; set; }
        // public BoundingBox BBox3D { get; set; }
        public int Handle { get; set; }
        //public GTAVector2 FUR;
        //public GTAVector2 FUL;
        //public GTAVector2 BUL;
        //public GTAVector2 BUR;
        //public GTAVector2 FLR;
        //public GTAVector2 FLL;
        //public GTAVector2 BLL;
        //public GTAVector2 BLR;
        //public GTAVector2 BBmin;
        //public GTAVector2 BBmax;
        public GTAVector FURGame;
        public GTAVector FULGame;
        public GTAVector BULGame;
        public GTAVector BURGame;
        public GTAVector FLRGame;
        public GTAVector FLLGame;
        public GTAVector BLLGame;
        public GTAVector BLRGame;

        public GTADetection(Entity e, DetectionType type, int ImgW, int ImgH, GTAVector CamPos)
        {
            Type = type.ToString();
            Pos = new GTAVector(e.Position);
            Distance = Game.Player.Character.Position.DistanceTo(new Vector3(Pos.X, Pos.Y, Pos.Z));

            //Unique identifier
            Handle = e.Handle;

            Rot = new GTAVector(e.Rotation);
            cls = "Unknown";
            Vector3 gmin;
            Vector3 gmax;
            e.Model.GetDimensions(out gmin, out gmax);
            Dim = new GTAVector(gmax - gmin);
            
            BoundingBox BBox3DGame = new SharpDX.BoundingBox((SharpDX.Vector3)new GTAVector(gmin), (SharpDX.Vector3)new GTAVector(gmax));
            FURGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[0].X, BBox3DGame.GetCorners()[0].Y, BBox3DGame.GetCorners()[0].Z)));
            FULGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[1].X, BBox3DGame.GetCorners()[1].Y, BBox3DGame.GetCorners()[1].Z)));
            BULGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[2].X, BBox3DGame.GetCorners()[2].Y, BBox3DGame.GetCorners()[2].Z)));
            BURGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[3].X, BBox3DGame.GetCorners()[3].Y, BBox3DGame.GetCorners()[3].Z)));
            FLRGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[4].X, BBox3DGame.GetCorners()[4].Y, BBox3DGame.GetCorners()[4].Z)));
            FLLGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[5].X, BBox3DGame.GetCorners()[5].Y, BBox3DGame.GetCorners()[5].Z)));
            BLLGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[6].X, BBox3DGame.GetCorners()[6].Y, BBox3DGame.GetCorners()[6].Z)));
            BLRGame = new GTAVector(e.GetOffsetInWorldCoords(new Vector3(BBox3DGame.GetCorners()[7].X, BBox3DGame.GetCorners()[7].Y, BBox3DGame.GetCorners()[7].Z)));

            // Solution with getting camera matrix from memory; but projection is not good!
            //GTAData.WorldToScreenRel(new Vector3(FURGame.X, FURGame.Y, FURGame.Z), out FUR);
            //GTAData.WorldToScreenRel(new Vector3(FULGame.X, FULGame.Y, FULGame.Z), out FUL);
            //GTAData.WorldToScreenRel(new Vector3(BULGame.X, BULGame.Y, BULGame.Z), out BUL);
            //GTAData.WorldToScreenRel(new Vector3(BURGame.X, BURGame.Y, BURGame.Z), out BUR);
            //GTAData.WorldToScreenRel(new Vector3(FLRGame.X, FLRGame.Y, FLRGame.Z), out FLR);
            //GTAData.WorldToScreenRel(new Vector3(FLLGame.X, FLLGame.Y, FLLGame.Z), out FLL);
            //GTAData.WorldToScreenRel(new Vector3(BLLGame.X, BLLGame.Y, BLLGame.Z), out BLL);
            //GTAData.WorldToScreenRel(new Vector3(BLRGame.X, BLRGame.Y, BLRGame.Z), out BLR);

            //Vector2 FURtmp = new Vector2(FUR.X, FUR.Y);
            //Vector2 FULtmp = new Vector2(FUL.X, FUL.Y);
            //Vector2 BULtmp = new Vector2(BUL.X, BUL.Y);
            //Vector2 BURtmp = new Vector2(BUR.X, BUR.Y);
            //Vector2 FLRtmp = new Vector2(FLR.X, FLR.Y);
            //Vector2 FLLtmp = new Vector2(FLL.X, FLL.Y);
            //Vector2 BLLtmp = new Vector2(BLL.X, BLL.Y);
            //Vector2 BLRtmp = new Vector2(BLR.X, BLR.Y);

            //// Calculate dimension of 2d bb
            //Vector2[] bb = GTAData.BB2D(new Vector2[]{ FURtmp, FULtmp, BULtmp, BURtmp, FLRtmp, FLLtmp, BLLtmp, BLRtmp });
            //BBmin = new GTAVector2(bb[0].X, bb[0].Y);
            //BBmax = new GTAVector2(bb[1].X, bb[1].Y);

            // Better use this 3Dto2D Transformation; but it is outsourced into python to save performance - unfortunately python does something weired, so we use this code
            Vector2 FURtmp = GTAData.get2Dfrom3D(new Vector3(FURGame.X, FURGame.Y, FURGame.Z), ImgW, ImgH);
            Vector2 FULtmp = GTAData.get2Dfrom3D(new Vector3(FULGame.X, FULGame.Y, FULGame.Z), ImgW, ImgH);
            Vector2 BULtmp = GTAData.get2Dfrom3D(new Vector3(BULGame.X, BULGame.Y, BULGame.Z), ImgW, ImgH);
            Vector2 BURtmp = GTAData.get2Dfrom3D(new Vector3(BURGame.X, BURGame.Y, BURGame.Z), ImgW, ImgH);
            Vector2 FLRtmp = GTAData.get2Dfrom3D(new Vector3(FLRGame.X, FLRGame.Y, FLRGame.Z), ImgW, ImgH);
            Vector2 FLLtmp = GTAData.get2Dfrom3D(new Vector3(FLLGame.X, FLLGame.Y, FLLGame.Z), ImgW, ImgH);
            Vector2 BLLtmp = GTAData.get2Dfrom3D(new Vector3(BLLGame.X, BLLGame.Y, BLLGame.Z), ImgW, ImgH);
            Vector2 BLRtmp = GTAData.get2Dfrom3D(new Vector3(BLRGame.X, BLRGame.Y, BLRGame.Z), ImgW, ImgH);

            //Calculate dimension of 2d bb
            //Vector2[] bb = GTAData.BB2D(new Vector2[] { FURtmp, FULtmp, BULtmp, BURtmp, FLRtmp, FLLtmp, BLLtmp, BLRtmp });

            // Scale points to img coords
            //FUR = GTAData.scalePoints(new Vector2(FURtmp.X, FURtmp.Y), ImgW, ImgH);
            //FUL = GTAData.scalePoints(new Vector2(FULtmp.X, FULtmp.Y), ImgW, ImgH);
            //BUL = GTAData.scalePoints(new Vector2(BULtmp.X, BULtmp.Y), ImgW, ImgH);
            //BUR = GTAData.scalePoints(new Vector2(BURtmp.X, BURtmp.Y), ImgW, ImgH);
            //FLR = GTAData.scalePoints(new Vector2(FLRtmp.X, FLRtmp.Y), ImgW, ImgH);
            //FLL = GTAData.scalePoints(new Vector2(FLLtmp.X, FLLtmp.Y), ImgW, ImgH);
            //BLL = GTAData.scalePoints(new Vector2(BLLtmp.X, BLLtmp.Y), ImgW, ImgH);
            //BLR = GTAData.scalePoints(new Vector2(BLRtmp.X, BLRtmp.Y), ImgW, ImgH);
            //BBmin = GTAData.scalePoints(new Vector2(bb[0].X, bb[0].Y), ImgW, ImgH);
            //BBmax = GTAData.scalePoints(new Vector2(bb[1].X, bb[1].Y), ImgW, ImgH);

            bool vis;
            int cat;
            GTAData.visibleOnScreen(new GTAVector[] { FURGame, BLLGame, FULGame, BURGame, BULGame, BLRGame, FLLGame, FLRGame }, e, CamPos, out vis, out cat);
            Visibility = vis;
            DistCat = cat;

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
        public float FrameTime { get; set; }
        public float StepTime { get; set; }
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

        public static Vector2[] BB2D(Vector2[] pts)
        {
            Vector2 Min = new Vector2(int.MaxValue, int.MaxValue);
            Vector2 Max = new Vector2(0, 0);

            foreach (Vector2 p in pts)
            {
                if (p.X < Min.X)
                    Min.X = p.X;
                if (p.X > Max.X)
                    Max.X = p.X;
                if (p.Y < Min.Y)
                    Min.Y = p.Y;
                if (p.Y > Max.Y)
                    Max.Y = p.Y;
            }

            if (Min.X < 0)
                Min.X = 0;
            if (Min.Y < 0)
                Min.Y = 0;

            if (Max.X > UI.WIDTH)
                Max.X = UI.WIDTH;
            if (Max.Y > UI.HEIGHT)
                Max.Y = UI.HEIGHT;

            return new Vector2[] { Min, Max };
        }

        public static GTAVector2 scalePoints(Vector2 p, int ImageWidth, int ImageHeight)
        {
            //return new GTAVector2((int)(p.X), (int)(p.Y));
            return new GTAVector2((int)(ImageWidth / (1.0 * UI.WIDTH) * p.X), (int)(ImageHeight / (1.0 * UI.HEIGHT) * p.Y));
            //return new GTAVector2(p.X, p.Y);
        }
    
        public static float getDotVectorResult(Vector3 camPos, Vector3 corner, Vector3 camForward)
        {            
            Vector3 dir = (corner - camPos).Normalized;
            float pos = Vector3.Dot(dir, camForward);

            if (SettingsReader.DEBUG_TRANS)
            {
                // Create a file to write to.
                using (StreamWriter file = File.AppendText(SettingsReader.DEBUG_PATH))
                {
                    file.WriteLine(corner.ToString() + "\t " + camPos.ToString() + "\t " + dir.ToString());
                    file.WriteLine("dirNorm: " + dir.ToString() + "\t pos: " + pos.ToString());
                }
            }

            if (pos >= 0)
            {
                // If vertices is in front of cam, the normalized distance will be printed
                return pos;
            }
            else
                // If vertices is behind of cam, -1 will be printed
                return -1.0f;
        }

        public static Vector2 get2Dfrom3D(Vector3 a, int ImageWidth, int ImageHeight)
        {
            if (SettingsReader.DEBUG_TRANS)
            {
                if (!File.Exists(SettingsReader.DEBUG_PATH))
                {
                    // Create a file to write to.
                    using (StreamWriter file = File.CreateText(SettingsReader.DEBUG_PATH))
                    {
                        file.WriteLine("All coordinates are in game space!");
                        file.WriteLine("");
                    }
                }
            }
            // http://orfe.princeton.edu/~alaink/SmartDrivingCars/Visitors/FilipowiczVideoGamesforAutonomousDriving.pdf
            // camera rotation 
            Vector3 theta = Vector3.Zero;
            theta = (float)(System.Math.PI / 180f) * World.RenderingCamera.Rotation;

            // camera direction, at 0 rotation the camera looks down the postive Y axis --> WorldNorth schaut somit immer in Cam-Richtung
            Vector3 camDir = rotate(Vector3.WorldNorth, theta);

            float sign = 1f;
            float scale = 1f;
            if (getDotVectorResult(World.RenderingCamera.Position, a, camDir) == -1)
            {
                sign = -1f;
                scale = 15f;
                camDir = sign * rotate(Vector3.WorldNorth, theta);
            }

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
            viewPlaneDist = viewPlaneDist * scale;
            // Punkt auf der Bildebene
            Vector3 viewPlanePointTMP = viewPlaneDist * viewerDistNorm + e;

            // move origin to upper left 
            Vector3 newOrigin = c + (viewWindowHeight / 2f) * camUp - (viewWindowWidth / 2f) * camEast;
            Vector3 viewPlanePoint = (viewPlanePointTMP + c) - newOrigin;

            float viewPlaneX = Vector3.Dot(viewPlanePoint, camEast) / Vector3.Dot(camEast, camEast);
            float viewPlaneZ = Vector3.Dot(viewPlanePoint, camUp) / Vector3.Dot(camUp, camUp);

            float screenX = viewPlaneX / viewWindowWidth * UI.WIDTH;
            float screenY = -viewPlaneZ / viewWindowHeight * UI.HEIGHT;

            // This part is for saving all the transformation steps to a txt for debugging

            if (SettingsReader.DEBUG_TRANS)
            {
                using (StreamWriter file = File.AppendText(SettingsReader.DEBUG_PATH))
                {
                    file.WriteLine("theta: " + theta.ToString());
                    file.WriteLine("camDir: " + camDir.ToString());
                    file.WriteLine("c: " + c.ToString());
                    file.WriteLine("e: " + e.ToString());
                    file.WriteLine("camUp: " + camUp.ToString());
                    file.WriteLine("camEast: " + camEast.ToString());
                    file.WriteLine("viewWindowHeight: " + viewWindowHeight.ToString());
                    file.WriteLine("viewWindowWidth: " + viewWindowWidth.ToString());
                    file.WriteLine("newOrigin: " + newOrigin.ToString());
                    file.WriteLine("del: " + del.ToString());
                    file.WriteLine("viewerDist: " + viewerDist.ToString());
                    file.WriteLine("viewerDistNorm: " + viewerDistNorm.ToString());
                    file.WriteLine("dot: " + dot.ToString());
                    file.WriteLine("ang: " + ang.ToString());
                    file.WriteLine("viewPlaneDist: " + viewPlaneDist.ToString());
                    file.WriteLine("viewPlanePointTMP: " + viewPlanePointTMP.ToString());
                    file.WriteLine("viewPlanePoint: " + viewPlanePoint.ToString());
                    file.WriteLine("viewPlaneX: " + viewPlaneX.ToString());
                    file.WriteLine("viewPlaneZ: " + viewPlaneZ.ToString());
                    file.WriteLine("screenX: " + screenX.ToString());
                    file.WriteLine("screenY: " + screenY.ToString());
                    file.WriteLine("########################################################");
                }
            }


            //return new Vector2((int)screenX, (int)screenY);
            return new Vector2(screenX, screenY);
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

        public static bool WorldToScreenRel(Vector3 entityPosition, out GTAVector2 screenCoords)
        {
            var mView = CameraHelper.GetCameraMatrix();
            mView.Transpose();

            var vForward = mView.Row4;
            var vRight = mView.Row2;
            var vUpward = mView.Row3;

            var result = new Vector3(0, 0, 0);
            result.Z = (vForward.X * entityPosition.X) + (vForward.Y * entityPosition.Y) + (vForward.Z * entityPosition.Z) + vForward.W;
            result.X = (vRight.X * entityPosition.X) + (vRight.Y * entityPosition.Y) + (vRight.Z * entityPosition.Z) + vRight.W;
            result.Y = (vUpward.X * entityPosition.X) + (vUpward.Y * entityPosition.Y) + (vUpward.Z * entityPosition.Z) + vUpward.W;

            if (result.Z < 0.001f)
            {
                screenCoords = new GTAVector2(0, 0);
                return false;
            }

            float invw = 1.0f / result.Z;
            result.X *= invw;
            result.Y *= invw;

            screenCoords =  new GTAVector2((int)(UI.WIDTH * 0.5f + result.X * UI.WIDTH * 0.5f), (int)(UI.HEIGHT * 0.5f - result.Y * UI.HEIGHT * 0.5f));
            //screenCoords = new Vector2(result.X, -result.Y);
            return true;
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

        public static bool visibleOnScreen(GTAVector[] vertices, Entity e, GTAVector CamPos, out bool visibile, out int distCat)
        {
            //sw.Restart();
            int cnt = 0;
            Vector3 f = new Vector3(CamPos.X, CamPos.Y, CamPos.Z);
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

            if (Vector3.Distance(f, e.Position) < 15f & cnt > 0)
            {
                distCat = 15;
                visibile = true;
                return true;
            }
            else if (Vector3.Distance(f, e.Position) < 50f & cnt > 1)
            {
                distCat = 50;
                visibile = true;
                return true;
            }
            else if (Vector3.Distance(f, e.Position) < 100f & cnt > 1)
            {
                distCat = 100;
                visibile = true;
                return true;
            }
            else if (Vector3.Distance(f, e.Position) < 150f & cnt > 2)
            {
                distCat = 150;
                visibile = true;
                return true;
            }
            else
            {
                distCat = 9999;
                visibile = false;
                return false;
            }
        }

        public static Vector3 RotationToDirection(Vector3 rotation)
        {
            var z = 0.01745329f * rotation.Z;
            var x = 0.01745329f * rotation.X;
            var num = Math.Abs(Math.Cos(x));
            return new Vector3
            {
                X = (float)(-Math.Sin(z) * num),
                Y = (float)(Math.Cos(z) * num),
                Z = (float)Math.Sin(x)
            };
        }

        public static GTAData DumpData(List<Weather> capturedWeathers)
        {
            Ped playerPed = Game.Player.Character;
            bool camActive;
            bool camActive2;
            Vector3 CamPos = World.RenderingCamera.Position;
            Vector3 CamRot = World.RenderingCamera.Rotation;
            if (!(World.RenderingCamera.Handle == -1))
            {
                if (playerPed.IsInVehicle())
                {
                    TestVehicle.TestVehicle.getActiveCam(out camActive);
                    if (camActive)
                        TestVehicle.TestVehicle.getCarMatrix(out CamPos, out CamRot);
                }
                else if (!(playerPed.IsInVehicle()))
                {
                    CamVision.CV.getActiveCam(out camActive2);
                    if (camActive2)
                    {
                        CamPos = playerPed.Position + Vector3.WorldUp * -1;
                        CamRot = World.RenderingCamera.Rotation;
                    }
                }
            }

            var ret = new GTAData();
            ret.CurrentWeather = World.Weather;
            ret.CapturedWeathers = capturedWeathers;
            
            ret.Timestamp = DateTime.UtcNow;
            ret.LocalTime = World.CurrentDayTime;
            ret.GameTime = Game.GameTime;
            ret.FrameTime = Game.LastFrameTime;
            ret.StepTime = Function.Call<float>((GTA.Native.Hash)0x0000000050597EE2);
            ret.CamPos = new GTAVector(CamPos);
            ret.CamRot = new GTAVector(CamRot);
            ret.CamDirection = new GTAVector(RotationToDirection(CamRot));
            ret.CamHash = World.RenderingCamera.GetHashCode();
            ret.CamFOV = World.RenderingCamera.FieldOfView;
            ret.CamNearClip = World.RenderingCamera.NearClip;
            ret.CamFarClip = World.RenderingCamera.FarClip;
            ret.ImageWidth = Game.ScreenResolution.Width;
            ret.ImageHeight = Game.ScreenResolution.Height;
            ret.UIWidth = UI.WIDTH;
            ret.UIHeight = UI.HEIGHT;
            ret.GamerPos = new GTAVector(playerPed.Position);

            if (Game.Player.Character.IsInVehicle())
            {
                ret.CarPos = new GTAVector(playerPed.CurrentVehicle.Position);
                ret.CarRot = new GTAVector(playerPed.CurrentVehicle.Rotation);
            }
            else
            {
                ret.CarPos = new GTAVector(Vector3.Zero);
                ret.CarRot = new GTAVector(Vector3.Zero);
            }
            
            var peds = World.GetNearbyPeds(Game.Player.Character,  SettingsReader.maxAnnotationRange);
            var cars = World.GetNearbyVehicles(Game.Player.Character, SettingsReader.maxAnnotationRange);
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