// Convolution: 1×1 or 3×3, stride 1 or 2, optional bias + activation
#include "AACommon.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint inHW = g_InHeight * g_InWidth;
    uint outHW = g_Height * g_Width;
    uint inBase = g_InBuf * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;
    uint ks = g_KernelSize;
    int pad = (int)(ks / 2);

    #define MAX_OUT_CH 32
    float sums[MAX_OUT_CH];
    for (uint oc = 0; oc < g_OutChannels; oc++) sums[oc] = 0.0;

    for (uint ic = 0; ic < g_InChannels; ic++)
    {
        for (uint ky = 0; ky < ks; ky++)
        for (uint kx = 0; kx < ks; kx++)
        {
            int iy = (int)(y * g_Stride) + (int)ky - pad;
            int ix = (int)(x * g_Stride) + (int)kx - pad;
            if (iy >= 0 && iy < (int)g_InHeight && ix >= 0 && ix < (int)g_InWidth)
            {
                float val = g_Features[inBase + ic * inHW + (uint)iy * g_InWidth + (uint)ix];
                uint wBase = g_WeightOff + ic * (ks * ks) + ky * ks + kx;
                for (uint oc = 0; oc < g_OutChannels; oc++)
                    sums[oc] += val * g_Weights[wBase + oc * (g_InChannels * ks * ks)];
            }
        }
    }

    for (uint oc2 = 0; oc2 < g_OutChannels; oc2++)
    {
        float s = sums[oc2];
        if (g_BiasOff != 0xFFFFFFFF) s += g_Weights[g_BiasOff + oc2];
        if (g_Activation != ACT_NONE) s = ApplyActivation(s, g_Activation);
        g_Features[outBase + oc2 * outHW + y * g_Width + x] = s;
    }
}
