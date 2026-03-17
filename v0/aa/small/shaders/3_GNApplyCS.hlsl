// GroupNorm apply: normalize + gamma/beta + activation + optional skip
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint inBase = g_InBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;
    uint chPerGroup = g_OutChannels / g_NumGroups;
    bool hasSkip = (g_Flags & FLAG_HAS_SKIP) != 0;
    uint skipBase = g_AuxBuf * g_BufStride;
    uint idx = y * g_Width + x;

    for (uint c = 0; c < g_OutChannels; c++)
    {
        uint grp = c / chPerGroup;
        float mean = g_GNStats[grp * 2 + 0];
        float invStd = g_GNStats[grp * 2 + 1];

        float val = g_Features[inBase + c * hw + idx];
        val = (val - mean) * invStd;
        val = val * g_Weights[g_GammaOff + c] + g_Weights[g_BetaOff + c];

        if (hasSkip)
            val += g_Features[skipBase + c * hw + idx];

        val = ApplyActivation(val, g_Activation);
        g_Features[outBase + c * hw + idx] = val;
    }
}
