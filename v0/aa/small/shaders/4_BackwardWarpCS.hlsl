// Backward warp: warp InBuf using MV from AuxBuf → OutBuf (bilinear)
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint srcBase = g_InBuf * g_BufStride;
    uint mvBase = g_AuxBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;
    uint idx = y * g_Width + x;

    float mv_x = g_Features[mvBase + 0 * hw + idx];
    float mv_y = g_Features[mvBase + 1 * hw + idx];

    float sampleX = (float)x + mv_x;
    float sampleY = (float)y + mv_y;

    sampleX = clamp(sampleX, 0.0, (float)g_Width - 1.001);
    sampleY = clamp(sampleY, 0.0, (float)g_Height - 1.001);

    int x0 = (int)floor(sampleX);
    int y0 = (int)floor(sampleY);
    int x1 = clamp(x0 + 1, 0, (int)g_Width - 1);
    int y1 = clamp(y0 + 1, 0, (int)g_Height - 1);
    x0 = clamp(x0, 0, (int)g_Width - 1);
    y0 = clamp(y0, 0, (int)g_Height - 1);

    float fx = sampleX - floor(sampleX);
    float fy = sampleY - floor(sampleY);
    float w00 = (1.0 - fx) * (1.0 - fy);
    float w10 = fx * (1.0 - fy);
    float w01 = (1.0 - fx) * fy;
    float w11 = fx * fy;

    for (uint c = 0; c < g_InChannels; c++)
    {
        float v00 = g_Features[srcBase + c * hw + (uint)y0 * g_Width + (uint)x0];
        float v10 = g_Features[srcBase + c * hw + (uint)y0 * g_Width + (uint)x1];
        float v01 = g_Features[srcBase + c * hw + (uint)y1 * g_Width + (uint)x0];
        float v11 = g_Features[srcBase + c * hw + (uint)y1 * g_Width + (uint)x1];

        g_Features[outBase + c * hw + idx] = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
    }
}
