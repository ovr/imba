// Specialized Attention: C=32, concat(curr[32], warp[32]) → 1×1 conv → sigmoid → blend
#include "AACommon.hlsli"

#define C 32

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint currBase = g_InBuf * g_BufStride;
    uint warpBase = g_AuxBuf * g_BufStride;
    uint outBase  = g_OutBuf * g_BufStride;
    uint idx = y * g_Width + x;

    // Debug mode 3: skip temporal fusion
    if (g_DebugMode == 3) {
        [unroll] for (uint oc = 0; oc < C; oc++)
            g_Features[outBase + oc * hw + idx] = g_Features[currBase + oc * hw + idx];
        return;
    }

    float sums[C];
    [unroll] for (uint oc = 0; oc < C; oc++) sums[oc] = 0.0;

    // Curr channels (weight columns 0..31)
    [unroll] for (uint ic = 0; ic < C; ic++)
    {
        float val = g_Features[currBase + ic * hw + idx];
        [unroll] for (uint oc = 0; oc < C; oc++)
            sums[oc] += val * g_Weights[g_WeightOff + oc * (C * 2) + ic];
    }

    // Warp channels (weight columns 32..63)
    [unroll] for (uint ic2 = 0; ic2 < C; ic2++)
    {
        float val = g_Features[warpBase + ic2 * hw + idx];
        [unroll] for (uint oc = 0; oc < C; oc++)
            sums[oc] += val * g_Weights[g_WeightOff + oc * (C * 2) + C + ic2];
    }

    // Sigmoid + blend
    [unroll] for (uint oc2 = 0; oc2 < C; oc2++)
    {
        float sum = sums[oc2] + g_Weights[g_BiasOff + oc2];
        float attn = 1.0 / (1.0 + exp(-sum));
        float curr_val = g_Features[currBase + oc2 * hw + idx];
        float warp_val = g_Features[warpBase + oc2 * hw + idx];
        g_Features[outBase + oc2 * hw + idx] = curr_val + warp_val * attn;
    }
}
