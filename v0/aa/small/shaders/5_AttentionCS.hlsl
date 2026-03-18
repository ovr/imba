// Temporal attention: concat(curr, warped) → 1×1 conv → sigmoid → blend
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint currBase = g_InBuf * g_BufStride;
    uint warpBase = g_AuxBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;
    uint idx = y * g_Width + x;
    uint C = g_InChannels; // 32

    // Debug mode 3: skip temporal fusion, pass current features only
    if (g_DebugMode == 3) {
        for (uint oc = 0; oc < C; oc++)
            g_Features[outBase + oc * hw + idx] = g_Features[currBase + oc * hw + idx];
        return;
    }

    #define MAX_ATT_CH 32
    float sums[MAX_ATT_CH];
    for (uint oc = 0; oc < C; oc++) sums[oc] = 0.0;

    for (uint ic = 0; ic < C * 2; ic++)
    {
        float val;
        if (ic < C)
            val = g_Features[currBase + ic * hw + idx];
        else
            val = g_Features[warpBase + (ic - C) * hw + idx];
        for (uint oc = 0; oc < C; oc++)
            sums[oc] += val * g_Weights[g_WeightOff + oc * (C * 2) + ic];
    }

    for (uint oc2 = 0; oc2 < C; oc2++)
    {
        float sum = sums[oc2] + g_Weights[g_BiasOff + oc2];
        float attn = 1.0 / (1.0 + exp(-sum));

        float curr_val = g_Features[currBase + oc2 * hw + idx];
        float warp_val = g_Features[warpBase + oc2 * hw + idx];
        g_Features[outBase + oc2 * hw + idx] = curr_val + warp_val * attn;
    }
}
