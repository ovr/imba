// GroupNorm final reduction: reduce 64 partial sums → mean/invStd per group
// Dispatch: (NumGroups, 1, 1), 64 threads per group
#include "AACommon.hlsli"

#define TILES_PER_GROUP 64

groupshared float2 gs_wave[2]; // max 2 waves for 64 threads

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint gnGroup = gid.x;
    if (gnGroup >= g_NumGroups) return;

    uint tid = gi;

    // Each thread reads one partial
    uint partialIdx = (gnGroup * TILES_PER_GROUP + tid) * 2;
    float partialSum = g_GNPartials[partialIdx + 0];
    float partialSumSq = g_GNPartials[partialIdx + 1];

    // Wave-level reduction
    float totalSum = WaveActiveSum(partialSum);
    float totalSumSq = WaveActiveSum(partialSumSq);

    // Cross-wave reduction (needed when wave size < 64, e.g. 32 on NVIDIA)
    uint waveIdx = tid / WaveGetLaneCount();
    uint laneIdx = WaveGetLaneIndex();

    if (laneIdx == 0)
        gs_wave[waveIdx] = float2(totalSum, totalSumSq);

    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        uint numWaves = (64 + WaveGetLaneCount() - 1) / WaveGetLaneCount();
        float finalSum = 0;
        float finalSumSq = 0;
        for (uint w = 0; w < numWaves; w++)
        {
            finalSum += gs_wave[w].x;
            finalSumSq += gs_wave[w].y;
        }

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
