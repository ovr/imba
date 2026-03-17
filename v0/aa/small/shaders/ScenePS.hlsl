cbuffer FrameConstants : register(b0)
{
    float4x4 ViewProj;
    float4x4 PrevViewProj;
    float2   Jitter;
    float2   Resolution;
    float    Time;
    uint     FrameIndex;
    float    TimeOfDay;
    float    _pad;
};

struct PSInput
{
    float4 Position     : SV_Position;
    float4 CurClip      : TEXCOORD0;
    float4 PrevClip     : TEXCOORD1;
    float3 Normal       : TEXCOORD2;
    float4 Color        : COLOR;
};

struct PSOutput
{
    float4 Color        : SV_Target0;   // R16G16B16A16_FLOAT HDR color
    float2 MotionVector : SV_Target1;   // R16G16_FLOAT motion vectors
};

// Compute sun direction from TimeOfDay (0..1, 0=midnight, 0.5=noon)
float3 GetSunDirection(float tod)
{
    float angle = tod * 6.28318530718; // full circle
    // Sun rises in east (+X), peaks overhead at noon, sets in west (-X)
    float elevation = -cos(angle);     // -1 at midnight, +1 at noon
    float horizontal = sin(angle);
    return normalize(float3(horizontal, elevation, 0.3));
}

PSOutput main(PSInput input)
{
    PSOutput output;

    float3 sunDir = GetSunDirection(TimeOfDay);
    float sunElevation = sunDir.y; // -1 to +1

    // Lighting intensity varies with sun elevation
    float dayFactor = saturate(sunElevation * 2.0 + 0.5); // 0 at night, 1 during day

    float3 N = normalize(input.Normal);
    float NdotL = saturate(dot(N, sunDir));

    // Sun color shifts: warm at low angles, white at noon
    float3 sunColor = lerp(float3(1.0, 0.5, 0.2), float3(1.0, 0.95, 0.9),
                           saturate(sunElevation));

    // Ambient: blue tint at night, warm during day
    float3 nightAmbient = float3(0.03, 0.04, 0.08);
    float3 dayAmbient   = float3(0.25, 0.25, 0.30);
    float3 ambient = lerp(nightAmbient, dayAmbient, dayFactor);

    float3 diffuse = NdotL * sunColor * 0.7 * dayFactor;

    output.Color = float4(input.Color.rgb * (ambient + diffuse), 1.0);

    // Motion vectors in UV space: (curUV - prevUV)
    float2 curNDC  = input.CurClip.xy / input.CurClip.w;
    float2 prevNDC = input.PrevClip.xy / input.PrevClip.w;
    float2 diff = curNDC - prevNDC;
    output.MotionVector = diff * float2(0.5, -0.5);

    return output;
}
