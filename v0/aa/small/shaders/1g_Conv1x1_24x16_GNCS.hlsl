// Fused Conv1x1 + GN partial accumulation: IN_CH=24, OUT_CH=16, stride=1
#include "AACommon.hlsli"

#define IN_CH  24
#define OUT_CH 16

#include "ConvGNPartials.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint x = dtid.x, y = dtid.y;

    float sums[OUT_CH];
    [unroll] for (uint oc = 0; oc < OUT_CH; oc++) sums[oc] = 0.0;

    bool valid = (x < g_Width && y < g_Height);

    if (valid)
    {
        uint hw = g_Height * g_Width;
        uint inBase  = g_InBuf  * g_BufStride;
        uint outBase = g_OutBuf * g_BufStride;
        uint pixIdx  = y * g_Width + x;

        [unroll] for (uint ic = 0; ic < IN_CH; ic++)
        {
            float val = g_Features[inBase + ic * hw + pixIdx];
            uint wOff = g_WeightOff + ic;
            [unroll] for (uint oc = 0; oc < OUT_CH; oc++)
                sums[oc] += val * g_Weights[wOff + oc * IN_CH];
        }

        [unroll] for (uint oc2 = 0; oc2 < OUT_CH; oc2++)
        {
            float s = sums[oc2];
            if (g_BiasOff != 0xFFFFFFFF) s += g_Weights[g_BiasOff + oc2];
            g_Features[outBase + oc2 * hw + y * g_Width + x] = s;
            sums[oc2] = s;
        }
    }

    AccumulateGNPartials(gid, gi, sums);
}
