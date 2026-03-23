// Scale motion vectors: bilinear downsample + UV→pixel conversion
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint outBase = g_OutBuf * g_BufStride;
    uint idx = y * g_Width + x;

    // Debug mode 2: zero motion vectors
    if (g_DebugMode == 2) {
        g_Features[outBase + 0 * hw + idx] = 0.0;
        g_Features[outBase + 1 * hw + idx] = 0.0;
        return;
    }

    float2 uv = (float2(x, y) + 0.5) / float2(g_Width, g_Height);
    float2 mv_uv = g_CurrMotion.SampleLevel(g_LinearSampler, uv, 0);

    // UV → pixel-space at quarter res, negate both for backward warp (sample = cur + mv)
    g_Features[outBase + 0 * hw + idx] = mv_uv.x * -(float)g_Width;
    g_Features[outBase + 1 * hw + idx] = mv_uv.y * -(float)g_Height;
}
