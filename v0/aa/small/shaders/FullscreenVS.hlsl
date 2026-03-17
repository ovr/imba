struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
};

// Full-screen triangle trick: 3 vertices, no vertex buffer
VSOutput main(uint vertexID : SV_VertexID)
{
    VSOutput output;

    // Generate full-screen triangle
    output.TexCoord = float2((vertexID << 1) & 2, vertexID & 2);
    output.Position = float4(output.TexCoord * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

    return output;
}
