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

    for (uint oc = 0; oc < g_OutChannels; oc++)
    {
        float sum = 0.0;
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
                    uint wIdx = g_WeightOff + oc * (g_InChannels * ks * ks) + ic * (ks * ks) + ky * ks + kx;
                    sum += val * g_Weights[wIdx];
                }
            }
        }
        if (g_BiasOff != 0xFFFFFFFF)
            sum += g_Weights[g_BiasOff + oc];
        if (g_Activation != ACT_NONE)
            sum = ApplyActivation(sum, g_Activation);

        g_Features[outBase + oc * outHW + y * g_Width + x] = sum;
    }
}
