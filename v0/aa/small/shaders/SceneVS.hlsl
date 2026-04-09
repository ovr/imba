cbuffer FrameConstants : register(b0)
{
    float4x4 ViewProj;
    float4x4 PrevViewProj;
    float2   Jitter;
    float2   PrevJitter;
    float2   Resolution;
    float    Time;
    uint     FrameIndex;
    float    TimeOfDay;
    float    _pad;
};

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
    float4 Color    : COLOR;
};

struct VSOutput
{
    float4 Position     : SV_Position;
    float4 CurClip      : TEXCOORD0;
    float4 PrevClip     : TEXCOORD1;
    float3 Normal       : TEXCOORD2;
    float3 WorldPos     : TEXCOORD3;
    float4 Color        : COLOR;
};

VSOutput main(VSInput input)
{
    VSOutput output;

    float4 worldPos = float4(input.Position, 1.0);

    // Current frame clip position with jitter
    float4 curClip = mul(ViewProj, worldPos);
    float4 jitteredClip = curClip;
    jitteredClip.xy += Jitter * curClip.w;

    // Previous frame clip position (no jitter)
    float4 prevClip = mul(PrevViewProj, worldPos);

    output.Position = jitteredClip;
    output.CurClip  = curClip;      // unjittered for motion vectors
    output.PrevClip = prevClip;
    output.Normal   = input.Normal;
    output.WorldPos = input.Position;
    output.Color    = input.Color;

    return output;
}
