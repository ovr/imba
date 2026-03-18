// GroupNorm partial reduction: compute partial sums per tile → g_GNPartials
// Dispatch: (NumGroups * TILES_PER_GROUP, 1, 1), 256 threads per group
// Pass B (2b_GNStatsReduceCS) finalizes mean/invStd from partials.
#include "AACommon.hlsli"

#define TILES_PER_GROUP 64

groupshared float2 gs_accum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint gnGroup = gid.x / TILES_PER_GROUP;
    uint tileIdx = gid.x % TILES_PER_GROUP;
    uint tid = gi;

    if (gnGroup >= g_NumGroups) return;

    uint chPerGroup = g_OutChannels / g_NumGroups;
    uint startCh = gnGroup * chPerGroup;
    uint hw = g_Height * g_Width;
    uint totalElements = chPerGroup * hw;
    uint bufBase = g_InBuf * g_BufStride;

    // Each tile handles a contiguous slice of totalElements
    uint elemsPerTile = (totalElements + TILES_PER_GROUP - 1) / TILES_PER_GROUP;
    uint tileStart = tileIdx * elemsPerTile;
    uint tileEnd = min(tileStart + elemsPerTile, totalElements);

    float localSum = 0.0;
    float localSumSq = 0.0;

    for (uint i = tileStart + tid; i < tileEnd; i += 256)
    {
        uint ch = startCh + i / hw;
        uint sp = i % hw;
        float val = g_Features[bufBase + ch * hw + sp];
        localSum += val;
        localSumSq += val * val;
    }

    // Wave-level reduction first
    float waveSum = WaveActiveSum(localSum);
    float waveSumSq = WaveActiveSum(localSumSq);

    uint laneIdx = WaveGetLaneIndex();
    uint waveIdx = tid / WaveGetLaneCount();

    if (laneIdx == 0)
        gs_accum[waveIdx] = float2(waveSum, waveSumSq);

    GroupMemoryBarrierWithGroupSync();

    // Final reduction across waves
    uint numWaves = (256 + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (tid < numWaves)
    {
        float2 v = gs_accum[tid];
        float finalSum = WaveActiveSum(v.x);
        float finalSumSq = WaveActiveSum(v.y);

        if (tid == 0)
        {
            uint partialIdx = (gnGroup * TILES_PER_GROUP + tileIdx) * 2;
            g_GNPartials[partialIdx + 0] = finalSum;
            g_GNPartials[partialIdx + 1] = finalSumSq;
        }
    }
}
