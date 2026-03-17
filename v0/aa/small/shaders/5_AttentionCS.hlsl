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

    for (uint oc = 0; oc < C; oc++)
    {
        float curr_val = g_Features[currBase + oc * hw + idx];

        // Debug mode 3: skip temporal fusion, pass current features only
        if (g_DebugMode == 3) {
            g_Features[outBase + oc * hw + idx] = curr_val;
            continue;
        }

        float sum = 0.0;
        for (uint ic = 0; ic < C * 2; ic++)
        {
            float val;
            if (ic < C)
                val = g_Features[currBase + ic * hw + idx];
            else
                val = g_Features[warpBase + (ic - C) * hw + idx];
            sum += val * g_Weights[g_WeightOff + oc * (C * 2) + ic];
        }
        sum += g_Weights[g_BiasOff + oc];
        float attn = 1.0 / (1.0 + exp(-sum));

        float warp_val = g_Features[warpBase + oc * hw + idx];
        g_Features[outBase + oc * hw + idx] = curr_val + warp_val * attn;
    }
}
