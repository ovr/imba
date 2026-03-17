// Placeholder upscale compute shader - bilinear upscale
// Will be replaced with ML upscaler

Texture2D<float4>   InputColor    : register(t0);
Texture2D<float>    InputDepth    : register(t1);
Texture2D<float2>   InputMotion   : register(t2);
Texture2D<float4>   PrevFrame     : register(t3);

RWTexture2D<float4> OutputColor   : register(u0);

SamplerState LinearSampler : register(s0);

cbuffer UpscaleConstants : register(b0)
{
    uint   InputWidth;
    uint   InputHeight;
    uint   OutputWidth;
    uint   OutputHeight;
    uint   FrameIndex;
    float  DeltaTime;
    float2 Jitter;
};

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= OutputWidth || DTid.y >= OutputHeight)
        return;

    // UV in output space
    float2 uv = (float2(DTid.xy) + 0.5) / float2(OutputWidth, OutputHeight);

    // Simple bilinear sample from low-res input
    float4 color = InputColor.SampleLevel(LinearSampler, uv, 0);

    // Placeholder: just bilinear upscale, no temporal accumulation.
    // Temporal logic will be handled by the ML model.
    OutputColor[DTid.xy] = color;
}
