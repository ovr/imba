// Skip-connection: concat InBuf + AuxBuf → 1×1 conv → OutBuf
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint inBase = g_InBuf * g_BufStride;
    uint auxBase = g_AuxBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;
    uint idx = y * g_Width + x;
    uint totalIn = g_InChannels;
    uint halfIn = totalIn / 2;

    #define MAX_OUT_CH 32
    float sums[MAX_OUT_CH];
    for (uint oc = 0; oc < g_OutChannels; oc++) sums[oc] = 0.0;

    for (uint ic = 0; ic < totalIn; ic++)
    {
        float val;
        if (ic < halfIn)
            val = g_Features[inBase + ic * hw + idx];
        else
            val = g_Features[auxBase + (ic - halfIn) * hw + idx];
        for (uint oc = 0; oc < g_OutChannels; oc++)
            sums[oc] += val * g_Weights[g_WeightOff + oc * totalIn + ic];
    }

    for (uint oc2 = 0; oc2 < g_OutChannels; oc2++)
    {
        float s = sums[oc2];
        if (g_BiasOff != 0xFFFFFFFF)
            s += g_Weights[g_BiasOff + oc2];
        g_Features[outBase + oc2 * hw + idx] = s;
    }
}
