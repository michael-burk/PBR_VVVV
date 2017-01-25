//@author: vux
//@help: standard constant shader
//@tags: color
//@credits: 

struct vsInput
{
    float4 posObject : POSITION;
};

struct psInput
{
    float4 posScreen : SV_Position;
	float4 posObject : POSITION;
};

SamplerState linearSampler <string uiname="Sampler State";>
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

cbuffer cbPerDraw : register(b0)
{
	float4x4 tVP : VIEWPROJECTION;
};

cbuffer cbPerObj : register( b1 )
{	
	float4x4 tW : WORLD;
	float3 lightPos;
	float2 nearFarPlane;
	float depthOffset;
	
};


psInput VS(vsInput input)
{
	psInput output;
	output.posObject = mul(input.posObject,tW);
	output.posScreen = mul(input.posObject,mul(tW,tVP));
	return output;
}



float4 PS(psInput input): SV_Target
{
    float4 col = 0;
	
	float worldSpaceDistance = distance(lightPos, input.posObject.xyz);
	float dist = (worldSpaceDistance - nearFarPlane.x) /
              (nearFarPlane.y - nearFarPlane.x) + depthOffset;

	col.r = saturate(dist);
	col.g = col.r * col.r;

    return col;
}



technique11 VSM
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_5_0, VS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS() ) );
	}
}





