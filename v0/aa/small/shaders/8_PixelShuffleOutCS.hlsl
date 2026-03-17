// PixelShuffle output: [12, H/2, W/2] → [3, H, W] + add to input color
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    // Debug mode 1 or 5: output origColor only (no residual)
    // Mode 1 skips all earlier dispatches; mode 5 runs the full pipeline first
    if (g_DebugMode == 1 || g_DebugMode == 5) {
        float3 origColor = g_CurrColor[int2(x, y)].rgb;
        g_Output[int2(x, y)] = float4(origColor, 1.0);
        return;
    }

    uint inBase = g_InBuf * g_BufStride;
    uint inW = g_InWidth;
    uint inH = g_InHeight;
    uint inHW = inH * inW;

    uint sub_x = x % 2;
    uint sub_y = y % 2;
    uint in_x = x / 2;
    uint in_y = y / 2;

    // Channel group: TL=0, TR=1, BL=2, BR=3
    uint chGroup = sub_y * 2 + sub_x;

    float3 residual;
    residual.r = g_Features[inBase + (chGroup * 3 + 0) * inHW + in_y * inW + in_x];
    residual.g = g_Features[inBase + (chGroup * 3 + 1) * inHW + in_y * inW + in_x];
    residual.b = g_Features[inBase + (chGroup * 3 + 2) * inHW + in_y * inW + in_x];

    float3 origColor = g_CurrColor[int2(x, y)].rgb;

    // Debug mode 6: visualize residual only (scaled for visibility)
    if (g_DebugMode == 6) {
        g_Output[int2(x, y)] = float4(residual * 5.0 + 0.5, 1.0);
        return;
    }

    g_Output[int2(x, y)] = float4(origColor + residual, 1.0);
}
