// GroupNorm statistics: compute mean/invStd per group → g_GNStats
// Dispatch: (NumGroups, 1, 1), 256 threads per group
#include "AACommon.hlsli"

groupshared float gs_sum[256];
groupshared float gs_sumSq[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint groupIdx = gid.x;
    uint tid = gi;

    if (groupIdx >= g_NumGroups) return;

    uint chPerGroup = g_OutChannels / g_NumGroups;
    uint startCh = groupIdx * chPerGroup;
    uint hw = g_Height * g_Width;
    uint totalElements = chPerGroup * hw;
    uint bufBase = g_InBuf * g_BufStride;

    float localSum = 0.0;
    float localSumSq = 0.0;

    for (uint i = tid; i < totalElements; i += 256)
    {
        uint ch = startCh + i / hw;
        uint spatial = i % hw;
        float val = g_Features[bufBase + ch * hw + spatial];
        localSum += val;
        localSumSq += val * val;
    }

    gs_sum[tid] = localSum;
    gs_sumSq[tid] = localSumSq;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction (256 → 1)
    [unroll] for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            gs_sum[tid] += gs_sum[tid + s];
            gs_sumSq[tid] += gs_sumSq[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        float mean = gs_sum[0] / (float)totalElements;
        float var = gs_sumSq[0] / (float)totalElements - mean * mean;
        float invStd = rsqrt(var + 1e-5);
        g_GNStats[groupIdx * 2 + 0] = mean;
        g_GNStats[groupIdx * 2 + 1] = invStd;
    }
}
