cbuffer FrameConstants : register(b0)
{
    float4x4 ViewProj;
    float4x4 PrevViewProj;
    float2   Jitter;
    float2   Resolution;
    float    Time;
    uint     FrameIndex;
    float    TimeOfDay;
    float    _pad0;
    float3   CameraPos;
    float    _pad1;
};

struct PSInput
{
    float4 Position     : SV_Position;
    float4 CurClip      : TEXCOORD0;
    float4 PrevClip     : TEXCOORD1;
    float3 Normal       : TEXCOORD2;
    float3 WorldPos     : TEXCOORD3;
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

// Procedural sky color for reflection
float3 SkyColor(float3 dir, float3 sunDir, float dayFactor)
{
    float y = dir.y;

    // Day sky gradient: horizon haze → deep blue at zenith
    float3 dayZenith  = float3(0.20, 0.35, 0.85);
    float3 dayHorizon = float3(0.55, 0.65, 0.80);
    float3 daySky = lerp(dayHorizon, dayZenith, saturate(y));

    // Night sky
    float3 nightSky = float3(0.01, 0.01, 0.03);

    float3 sky = lerp(nightSky, daySky, dayFactor);

    // Sun glow in reflection
    float sunDot = saturate(dot(dir, sunDir));
    float3 sunGlow = pow(sunDot, 64.0) * float3(1.0, 0.9, 0.7) * 2.0 * dayFactor;
    // Broader halo
    float3 sunHalo = pow(sunDot, 8.0) * float3(0.3, 0.25, 0.15) * dayFactor;

    return sky + sunGlow + sunHalo;
}

// Schlick Fresnel approximation for dielectric (glass IOR ~1.5)
float FresnelSchlick(float cosTheta)
{
    float R0 = 0.04; // ((1.5-1)/(1.5+1))^2
    return R0 + (1.0 - R0) * pow(1.0 - saturate(cosTheta), 5.0);
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

    float3 baseColor = input.Color.rgb * (ambient + diffuse);

    // Glass reflection when alpha < 1.0
    float alpha = input.Color.a;
    if (alpha < 0.99)
    {
        float3 V = normalize(CameraPos - input.WorldPos);
        float NdotV = dot(N, V);

        // Flip normal if facing away from camera (back-face of glass)
        if (NdotV < 0.0)
        {
            N = -N;
            NdotV = -NdotV;
        }

        float fresnel = FresnelSchlick(NdotV);

        // Boost fresnel: stronger during day for sky reflections
        float fresnelBoost = lerp(0.25, 0.55, dayFactor);
        fresnel = lerp(fresnel, 1.0, fresnelBoost);

        float3 reflDir = reflect(-V, N);
        float3 reflColor = SkyColor(reflDir, sunDir, dayFactor);

        // Interior color: the vertex color represents what's behind the glass
        float3 interiorColor = input.Color.rgb;

        // Day: reflections dominate heavily; night: interior glows through
        float reflStrength = lerp(0.35, 0.92, dayFactor);

        // Blend: fresnel controls reflection vs interior visibility
        float3 glassColor = lerp(interiorColor, reflColor, fresnel * reflStrength);

        // Specular highlight from sun — stronger during day
        float3 H = normalize(V + sunDir);
        float spec = pow(saturate(dot(N, H)), 96.0);
        glassColor += spec * sunColor * 0.8 * dayFactor;

        // Tint glass slightly blue-green
        glassColor *= float3(0.92, 0.96, 1.0);

        output.Color = float4(glassColor, 1.0);
    }
    else
    {
        output.Color = float4(baseColor, 1.0);
    }

    // Motion vectors in UV space: (curUV - prevUV)
    float2 curNDC  = input.CurClip.xy / input.CurClip.w;
    float2 prevNDC = input.PrevClip.xy / input.PrevClip.w;
    float2 diff = curNDC - prevNDC;
    output.MotionVector = diff * float2(0.5, -0.5);

    return output;
}
