Texture2D<float4> InputTexture : register(t0);
SamplerState       LinearSampler : register(s0);

cbuffer BlitConstants : register(b0)
{
    uint   VisualizationMode; // 0=color, 1=depth, 2=motion vectors
    float  Exposure;
    float  TimeOfDay;         // 0..1 (0=midnight, 0.5=noon)
    uint   HDREnabled;
};

Texture2D<float>  DepthTexture  : register(t1);
Texture2D<float2> MotionTexture : register(t2);

struct PSInput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
};

// ACES filmic tonemap
float3 ACESFilm(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Sun direction from TimeOfDay (must match ScenePS)
float3 GetSunDirection(float tod)
{
    float angle = tod * 6.28318530718;
    float elevation = -cos(angle);
    float horizontal = sin(angle);
    return normalize(float3(horizontal, elevation, 0.3));
}

// Simple hash for star noise
float Hash(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float4 main(PSInput input) : SV_Target
{
    if (VisualizationMode == 1)
    {
        float depth = DepthTexture.Sample(LinearSampler, input.TexCoord);
        float linearZ = 0.1 / (1.0 - depth + 0.001);
        float vis = saturate(linearZ / 100.0);
        return float4(vis, vis, vis, 1.0);
    }
    else if (VisualizationMode == 2)
    {
        float2 mv = MotionTexture.Sample(LinearSampler, input.TexCoord);
        float magnitude = length(mv) * 20.0;
        return float4(abs(mv.x) * 10.0, abs(mv.y) * 10.0, magnitude, 1.0);
    }

    float3 hdr = InputTexture.Sample(LinearSampler, input.TexCoord).rgb;

    float depth = DepthTexture.Sample(LinearSampler, input.TexCoord);
    if (depth >= 1.0)
    {
        // === Dynamic sky ===
        float3 sunDir = GetSunDirection(TimeOfDay);
        float sunElev = sunDir.y; // -1..+1
        float dayFactor = saturate(sunElev * 2.0 + 0.5); // 0 at night, 1 during day

        float2 uv = input.TexCoord;
        float elevation = 1.0 - uv.y; // 0=horizon, 1=zenith

        // Sky gradient changes with time of day
        // Night
        float3 nightHorizon = float3(0.02, 0.02, 0.05);
        float3 nightZenith  = float3(0.01, 0.01, 0.03);
        // Day
        float3 dayHorizon   = float3(0.7, 0.75, 0.85);
        float3 dayZenith    = float3(0.15, 0.25, 0.65);
        // Sunrise/sunset
        float3 dawnHorizon  = float3(0.9, 0.4, 0.15);
        float3 dawnZenith   = float3(0.2, 0.15, 0.35);

        // Blend factor for sunrise/sunset glow
        float dawnFactor = smoothstep(0.0, 0.3, sunElev) * smoothstep(0.6, 0.3, sunElev);

        float3 horizonColor = lerp(nightHorizon, dayHorizon, dayFactor);
        horizonColor = lerp(horizonColor, dawnHorizon, dawnFactor);
        float3 zenithColor = lerp(nightZenith, dayZenith, dayFactor);
        zenithColor = lerp(zenithColor, dawnZenith, dawnFactor * 0.5);

        float t = saturate(pow(elevation, 0.6));
        float3 sky = lerp(horizonColor, zenithColor, t);

        // Sun disk (only when above horizon)
        float2 ndc = uv * 2.0 - 1.0;
        ndc.y = -ndc.y;
        float3 viewDir = normalize(float3(ndc.x * 1.5, ndc.y, -1.0));
        float sunDot = dot(viewDir, sunDir);

        if (sunElev > -0.1)
        {
            float sunVis = saturate((sunElev + 0.1) * 5.0); // fade in near horizon

            // Sun color: orange at dawn/dusk, white at noon
            float3 sunColor = lerp(float3(1.5, 0.6, 0.2), float3(1.5, 1.3, 0.9),
                                   saturate(sunElev));

            // Sun core
            float sunMask = smoothstep(0.997, 0.999, sunDot);
            sky += sunMask * sunColor * sunVis;

            // Sun glow
            float glowMask = smoothstep(0.9, 0.999, sunDot);
            sky += glowMask * sunColor * 0.2 * sunVis;
        }

        // Atmospheric haze near horizon
        float hazeMask = 1.0 - saturate(elevation * 3.0);
        float3 hazeColor = lerp(float3(0.02, 0.02, 0.04), float3(0.15, 0.1, 0.05), dayFactor);
        hazeColor = lerp(hazeColor, float3(0.3, 0.15, 0.05), dawnFactor);
        sky += hazeMask * hazeColor;

        // Stars at night
        if (dayFactor < 0.8)
        {
            float starFade = 1.0 - smoothstep(0.0, 0.4, dayFactor);
            float2 starUV = uv * 200.0;
            float2 starCell = floor(starUV);
            float starNoise = Hash(starCell);

            if (starNoise > 0.97) // sparse stars
            {
                float2 cellCenter = (starCell + 0.5) / 200.0;
                float dist = length(uv - cellCenter) * 200.0;
                float starBright = smoothstep(0.5, 0.0, dist) * (starNoise - 0.97) * 33.0;
                // Slight twinkle
                float twinkle = sin(starNoise * 100.0 + TimeOfDay * 50.0) * 0.3 + 0.7;
                sky += starBright * starFade * twinkle * float3(0.8, 0.85, 1.0);
            }

            // Moon (opposite side from sun)
            float3 moonDir = -sunDir;
            moonDir.y = abs(moonDir.y) * 0.8; // keep moon above horizon
            moonDir = normalize(moonDir);
            float moonDot = dot(viewDir, moonDir);
            float moonMask = smoothstep(0.9995, 0.9998, moonDot);
            sky += moonMask * float3(0.5, 0.5, 0.6) * starFade;
            // Moon glow
            float moonGlow = smoothstep(0.98, 0.9998, moonDot);
            sky += moonGlow * float3(0.05, 0.05, 0.08) * starFade;
        }

        if (HDREnabled)
            return float4(sky * Exposure, 1.0);
        sky = ACESFilm(sky);
        sky = pow(sky, 1.0 / 2.2);
        return float4(sky, 1.0);
    }

    hdr *= Exposure;
    if (HDREnabled)
        return float4(hdr, 1.0);
    float3 ldr = ACESFilm(hdr);
    ldr = pow(ldr, 1.0 / 2.2);

    return float4(ldr, 1.0);
}
