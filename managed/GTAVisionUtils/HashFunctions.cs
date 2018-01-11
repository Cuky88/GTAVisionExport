using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GTA;
using GTA.Math;
using GTA.Native;

namespace GTAVisionUtils
{
    public class HashFunctions
    {
        public static Vector2 Convert3dTo2d(Vector3 pos)
        {
            OutputArgument tmpResX = new OutputArgument();
            OutputArgument tmpResY = new OutputArgument();
            if (Function.Call<bool>(Hash._WORLD3D_TO_SCREEN2D, (InputArgument) pos.X,
                (InputArgument) pos.Y, (InputArgument) pos.Z,
                (InputArgument) tmpResX, (InputArgument) tmpResY))
            {
                Vector2 v2;
                v2.X = tmpResX.GetResult<float>();
                v2.Y = tmpResY.GetResult<float>();
                return v2;
            }
            return new Vector2(-1f, -1f);
        }
        public static void Draw3DLine(Vector3 iniPos, Vector3 finPos, byte col_r = 255, byte col_g = 255, byte col_b = 255, byte col_a = 255) {
            Function.Call(Hash.DRAW_LINE, new InputArgument[]
            {
                iniPos.X,
                iniPos.Y,
                iniPos.Z,
                finPos.X,
                finPos.Y,
                finPos.Z,
                (int)col_r,
                (int)col_g,
                (int)col_b,
                (int)col_a
            });
        }
        public static void DrawRect(float x, float y, float w, float h, byte r = 255, byte g = 255, byte b = 255, byte a = 255) {
            Function.Call(Hash.DRAW_RECT, new InputArgument[] {
                x, y,
                w, h,
                (int)r, (int)g, (int)b, (int)a
            });
        }

        public static void ClearAreaOfVehicle(Vector3 pos, float distance, bool p1, bool p2, bool p3, bool p4)
        {
            Function.Call(GTA.Native.Hash.CLEAR_AREA_OF_VEHICLES, new InputArgument[] {
                pos.X,
                pos.Y,
                pos.Z,
                distance,
                p1,
                p2,
                p3,
                p4
            });
        }

        public static bool LOS(Ped ped, Entity e)
        {
            return Function.Call<bool>((GTA.Native.Hash)0x0267D00AF114F17A, ped, e);
        }

        public static void SpecialAbilityFillMeter(Player player, bool p1)
        {
            Function.Call(Hash.SPECIAL_ABILITY_FILL_METER, new InputArgument[]
            {
                player,
                p1
            });
        }

        public static void SetPlayerNoiseMultiplier(Player player, bool p1)
        {
            Function.Call(Hash.SET_PLAYER_NOISE_MULTIPLIER, new InputArgument[]
            {
                player,
                p1
            });
        }

        public static void SetPedConfigFlag(Ped ped, int flagId, bool p1)
        {
            Function.Call(Hash.SET_PED_CONFIG_FLAG, new InputArgument[]
            {
                ped,
                flagId,
                p1
            });
        }

        public static void SetCreateRandomCops(bool p1)
        {
            Function.Call(Hash.SET_CREATE_RANDOM_COPS, new InputArgument[]
            {
                p1
            });
        }

        public static void SetRandomTrains(bool p1)
        {
            Function.Call(Hash.SET_RANDOM_TRAINS, new InputArgument[]
            {
                p1
            });
        }

        public static void SetRandomBoats(bool p1)
        {
            Function.Call(Hash.SET_RANDOM_BOATS, new InputArgument[]
            {
                p1
            });
        }

        public static void SetGarbageTrucks(bool p1)
        {
            Function.Call(Hash.SET_GARBAGE_TRUCKS, new InputArgument[]
            {
                p1
            });
        }

    }
}
