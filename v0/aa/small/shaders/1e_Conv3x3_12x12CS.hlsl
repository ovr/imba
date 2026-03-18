// Specialized Conv3x3: IN_CH=12, OUT_CH=12, stride=1, pad=1
#include "AACommon.hlsli"

#define IN_CH  12
#define OUT_CH 12

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint x = dtid.x, y = dtid.y;
    if (x >= g_Width || y >= g_Height) return;

    uint inHW  = g_Height * g_Width;  // stride=1 so inH==outH, inW==outW
    uint outHW = inHW;
    uint inBase  = g_InBuf  * g_BufStride;
    uint outBase = g_OutBuf * g_BufStride;

    float sums[OUT_CH];
    [unroll] for (uint oc = 0; oc < OUT_CH; oc++) sums[oc] = 0.0;

    [unroll] for (uint ic = 0; ic < IN_CH; ic++)
    {
        [unroll] for (uint ky = 0; ky < 3; ky++)
        [unroll] for (uint kx = 0; kx < 3; kx++)
        {
            int iy = (int)y + (int)ky - 1;
            int ix = (int)x + (int)kx - 1;
            if (iy >= 0 && iy < (int)g_Height && ix >= 0 && ix < (int)g_Width)
            {
                float val = g_Features[inBase + ic * inHW + (uint)iy * g_Width + (uint)ix];
                uint wBase = g_WeightOff + ic * 9 + ky * 3 + kx;
                [unroll] for (uint oc = 0; oc < OUT_CH; oc++)
                    sums[oc] += val * g_Weights[wBase + oc * (IN_CH * 9)];
            }
        }
    }

    [unroll] for (uint oc2 = 0; oc2 < OUT_CH; oc2++)
    {
        float s = sums[oc2];
        if (g_BiasOff != 0xFFFFFFFF) s += g_Weights[g_BiasOff + oc2];
        if (g_Activation != ACT_NONE) s = ApplyActivation(s, g_Activation);
        g_Features[outBase + oc2 * outHW + y * g_Width + x] = s;
    }
}
