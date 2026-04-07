// Shared code for fused Conv + GN partial accumulation.
// Include AFTER computing sums[OUT_CH] and writing to output buffer.
// Requires: OUT_CH, g_NumGroups, g_Width, g_Height, g_NumTiles defined.
// Requires: float sums[OUT_CH] in scope with final conv output values.

#ifndef CONV_GN_PARTIALS_HLSLI
#define CONV_GN_PARTIALS_HLSLI

#define MAX_GN_GROUPS 4

groupshared float gs_gnSum[MAX_GN_GROUPS];
groupshared float gs_gnSumSq[MAX_GN_GROUPS];

void AccumulateGNPartials(uint3 gid, uint gi, float sums[OUT_CH])
{
    // Out-of-bounds threads contribute zero (already filtered by caller)
    uint chPerGroup = OUT_CH / g_NumGroups;

    // Each thread accumulates per-group sum/sumSq from its pixel's channels
    float tSum[MAX_GN_GROUPS];
    float tSumSq[MAX_GN_GROUPS];
    [unroll] for (uint g = 0; g < MAX_GN_GROUPS; g++)
    {
        tSum[g] = 0;
        tSumSq[g] = 0;
    }

    [unroll] for (uint c = 0; c < OUT_CH; c++)
    {
        uint grp = c / chPerGroup;
        tSum[grp] += sums[c];
        tSumSq[grp] += sums[c] * sums[c];
    }

    // Wave-level reduction per group
    [unroll] for (uint g2 = 0; g2 < MAX_GN_GROUPS; g2++)
    {
        float wSum   = WaveActiveSum(tSum[g2]);
        float wSumSq = WaveActiveSum(tSumSq[g2]);

        // First lane of each wave writes to groupshared
        if (WaveGetLaneIndex() == 0)
        {
            // For wave64: single wave covers all 64 threads, no cross-wave needed
            // For wave32: 2 waves, need atomic-style accumulation
            uint waveIdx = gi / WaveGetLaneCount();
            if (waveIdx == 0)
            {
                gs_gnSum[g2] = wSum;
                gs_gnSumSq[g2] = wSumSq;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Cross-wave reduction (only needed if wave size < 64)
    if (WaveGetLaneCount() < 64 && gi >= WaveGetLaneCount() && gi < WaveGetLaneCount() * 2)
    {
        uint laneInWave = gi - WaveGetLaneCount();
        if (laneInWave < MAX_GN_GROUPS)
        {
            float wSum   = WaveActiveSum(tSum[laneInWave]);
            float wSumSq = WaveActiveSum(tSumSq[laneInWave]);
            if (WaveGetLaneIndex() == 0)
            {
                gs_gnSum[laneInWave] += wSum;
                gs_gnSumSq[laneInWave] += wSumSq;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Thread 0 writes tile partials to gnPartials buffer
    if (gi == 0)
    {
        uint tilesX = (g_Width + 7) / 8;
        uint tileId = gid.y * tilesX + gid.x;

        [unroll] for (uint g3 = 0; g3 < MAX_GN_GROUPS; g3++)
        {
            if (g3 < g_NumGroups)
            {
                uint idx = (g3 * g_NumTiles + tileId) * 2;
                g_GNPartials[idx + 0] = gs_gnSum[g3];
                g_GNPartials[idx + 1] = gs_gnSumSq[g3];
            }
        }
    }
}

#endif // CONV_GN_PARTIALS_HLSLI
