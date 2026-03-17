// Nearest-neighbor 2× upsample
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint inHW = g_InHeight * g_InWidth;
    uint outHW = g_Height * g_Width;
    uint inBase = g_InBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;

    uint sx = min(x / 2, g_InWidth - 1);
    uint sy = min(y / 2, g_InHeight - 1);

    for (uint c = 0; c < g_InChannels; c++)
    {
        float val = g_Features[inBase + c * inHW + sy * g_InWidth + sx];
        g_Features[outBase + c * outHW + y * g_Width + x] = val;
    }
}
