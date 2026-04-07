// GroupNorm tiled reduction: reduce per-tile partials → mean/invStd per group
// Used with fused Conv+GN shaders that produce one partial per 8x8 tile.
// Dispatch: (NumGroups, 1, 1), 256 threads per group
#include "AACommon.hlsli"

groupshared float2 gs_accum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint gnGroup = gid.x;
    if (gnGroup >= g_NumGroups) return;

    uint numTiles = g_NumTiles;
    uint tid = gi;

    // Each thread sums a strided subset of tile partials
    float localSum = 0.0;
    float localSumSq = 0.0;

    for (uint t = tid; t < numTiles; t += 256)
    {
        uint idx = (gnGroup * numTiles + t) * 2;
        localSum += g_GNPartials[idx + 0];
        localSumSq += g_GNPartials[idx + 1];
    }

    // Wave-level reduction
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
            uint chPerGroup = g_OutChannels / g_NumGroups;
            uint hw = g_Height * g_Width;
            float totalElements = (float)(chPerGroup * hw);

            float mean = finalSum / totalElements;
            float var = finalSumSq / totalElements - mean * mean;
            float invStd = rsqrt(var + 1e-5);

            g_GNStats[gnGroup * 2 + 0] = mean;
            g_GNStats[gnGroup * 2 + 1] = invStd;
        }
    }
}
