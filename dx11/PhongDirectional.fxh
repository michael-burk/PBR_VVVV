
//phong directional function
lightStruct PhongDirectional(float3 NormV, float3 ViewDirV, float3 LightDirV, float4 lAmb, float4 lDiff,
						float4 lSpec, float4 specIntensity, float4 sF, lightStruct li)
{
    //halfvector
    float3 H = normalize(ViewDirV + LightDirV);

    //compute blinn lighting
    float3 shades = lit(dot(NormV, LightDirV), dot(NormV, H), lPower).xyz;

    float4 diff = lDiff * shades.y;
    diff.a = 1;

    //reflection vector (view space)
    float3 R = normalize(2 * dot(NormV, LightDirV) * NormV - LightDirV);

    //normalized view direction (view space)
    float3 V = normalize(ViewDirV);

    //calculate specular light
    float4 spec = pow(max(dot(R, V),0), lPower*.2) * lSpec;

    spec = spec * specIntensity;
	
	li.ambient += lAmb;
	li.diffuse += diff * sF;
	li.reflection += spec * sF;

	
    return li;
}