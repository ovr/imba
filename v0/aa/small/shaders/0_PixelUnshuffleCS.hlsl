// PixelUnshuffle: input textures → [32, H/2, W/2] feature buffer
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    bool isPrev = (g_Flags & FLAG_IS_PREV) != 0;
    float jx = isPrev ? g_PrevJitterX : g_JitterX;
    float jy = isPrev ? g_PrevJitterY : g_JitterY;

    // 4 sub-pixels: TL(0,0) TR(0,1) BL(1,0) BR(1,1)
    [unroll] for (uint sy = 0; sy < 2; sy++)
    [unroll] for (uint sx = 0; sx < 2; sx++)
    {
        int2 texCoord = int2(x * 2 + sx, y * 2 + sy);
        float3 color;
        float  depth;
        float2 motion;

        if (isPrev)
        {
            color  = g_PrevColor[texCoord].rgb;
            depth  = g_PrevDepth[texCoord];
            motion = g_PrevMotion[texCoord];
        }
        else
        {
            color  = g_CurrColor[texCoord].rgb;
            depth  = g_CurrDepth[texCoord];
            motion = g_CurrMotion[texCoord];
        }

        // UV → pixel-space, negate both for backward warp (sample = cur + mv)
        motion.x *= -(float)(g_InWidth);
        motion.y *= -(float)(g_InHeight);

        uint chBase = (sy * 2 + sx) * 8;
        uint oidx = y * g_Width + x;
        uint hw = g_Height * g_Width;
        uint base = g_OutBuf * g_BufStride;

        g_Features[base + (chBase + 0) * hw + oidx] = color.r;
        g_Features[base + (chBase + 1) * hw + oidx] = color.g;
        g_Features[base + (chBase + 2) * hw + oidx] = color.b;
        g_Features[base + (chBase + 3) * hw + oidx] = depth;
        g_Features[base + (chBase + 4) * hw + oidx] = motion.x;
        g_Features[base + (chBase + 5) * hw + oidx] = motion.y;
        g_Features[base + (chBase + 6) * hw + oidx] = jx;
        g_Features[base + (chBase + 7) * hw + oidx] = jy;
    }
}
