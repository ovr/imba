// Shared definitions for all AA compute passes.

#ifndef AA_COMMON_HLSLI
#define AA_COMMON_HLSLI

#define ACT_NONE    0
#define ACT_SILU    1
#define ACT_SIGMOID 2
#define ACT_TANH    3

#define FLAG_HAS_SKIP 1
#define FLAG_IS_PREV  2

cbuffer RootConstants : register(b0)
{
    uint g_PassType;     // 0  (unused per-file, kept for layout compat)
    uint g_InBuf;        // 1
    uint g_OutBuf;       // 2
    uint g_AuxBuf;       // 3
    uint g_Width;        // 4  (output spatial width)
    uint g_Height;       // 5  (output spatial height)
    uint g_InChannels;   // 6
    uint g_OutChannels;  // 7
    uint g_KernelSize;   // 8
    uint g_Stride;       // 9
    uint g_WeightOff;    // 10
    uint g_BiasOff;      // 11
    uint g_GammaOff;     // 12
    uint g_BetaOff;      // 13
    uint g_NumGroups;    // 14
    uint g_Activation;   // 15
    uint g_Flags;        // 16
    uint g_InWidth;      // 17 (input spatial width, may differ for stride/upsample)
    uint g_InHeight;     // 18
    uint g_BufStride;    // 19 (elements between feature buffer slots)
    float g_JitterX;     // 20
    float g_JitterY;     // 21
    float g_PrevJitterX; // 22
    float g_PrevJitterY; // 23
    uint g_DebugMode;    // 24
    uint _pad0;          // 25
    uint _pad1;          // 26
    uint _pad2;          // 27
};

StructuredBuffer<float> g_Weights : register(t0);
Texture2D<float4> g_CurrColor  : register(t1);
Texture2D<float>  g_CurrDepth  : register(t2);
Texture2D<float2> g_CurrMotion : register(t3);
Texture2D<float4> g_PrevColor  : register(t4);
Texture2D<float>  g_PrevDepth  : register(t5);
Texture2D<float2> g_PrevMotion : register(t6);

RWStructuredBuffer<float> g_Features : register(u0);
RWStructuredBuffer<float> g_GNStats  : register(u1);
RWTexture2D<float4> g_Output        : register(u2);

SamplerState g_LinearSampler : register(s0);

float ApplyActivation(float x, uint act)
{
    if (act == ACT_SILU)    return x / (1.0 + exp(-x));
    if (act == ACT_SIGMOID) return 1.0 / (1.0 + exp(-x));
    if (act == ACT_TANH)    return tanh(x);
    return x;
}

#endif // AA_COMMON_HLSLI
