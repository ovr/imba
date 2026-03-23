// Specialized SkipConcatConv: concat(in[16], aux[16]) → 1×1 conv → out[16], no bias
#include "AACommon.hlsli"

#define HALF_IN  16
#define OUT_CH   16
#define TOTAL_IN 32

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint hw = g_Height * g_Width;
    uint inBase  = g_InBuf  * g_BufStride;
    uint auxBase = g_AuxBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;
    uint idx = y * g_Width + x;

    float sums[OUT_CH];
    [unroll] for (uint oc = 0; oc < OUT_CH; oc++) sums[oc] = 0.0;

    // InBuf channels (weight columns 0..15)
    [unroll] for (uint ic = 0; ic < HALF_IN; ic++)
    {
        float val = g_Features[inBase + ic * hw + idx];
        [unroll] for (uint oc = 0; oc < OUT_CH; oc++)
            sums[oc] += val * g_Weights[g_WeightOff + oc * TOTAL_IN + ic];
    }

    // AuxBuf channels (weight columns 16..31)
    [unroll] for (uint ic2 = 0; ic2 < HALF_IN; ic2++)
    {
        float val = g_Features[auxBase + ic2 * hw + idx];
        [unroll] for (uint oc = 0; oc < OUT_CH; oc++)
            sums[oc] += val * g_Weights[g_WeightOff + oc * TOTAL_IN + HALF_IN + ic2];
    }

    // Output (no bias, no activation)
    [unroll] for (uint oc2 = 0; oc2 < OUT_CH; oc2++)
        g_Features[outBase + oc2 * hw + idx] = sums[oc2];
}
