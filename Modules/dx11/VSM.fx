//@author: vux
//@help: standard constant shader
//@tags: color
//@credits: 
Texture2D inputTexture <string uiname="Alpha Tex";>;

struct vsInput
{
    float4 posObject : POSITION;
};

struct psInput
{
    float4 posScreen : SV_Position;
	float4 posObject : POSITION;

};

struct vsInput_AT
{
    float4 posObject : POSITION;
	float4 uv: TEXCOORD0;
};

struct psInput_AT
{
    float4 posScreen : SV_Position;
	float4 posObject : POSITION;
	float4 uv: TEXCOORD0;
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

cbuffer cbTextureData : register(b2)
{
	float4x4 tTex <string uiname="Texture Transform"; bool uvspace=true; >;
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

psInput_AT VS_AT(vsInput_AT input)
{
	psInput_AT output;
	output.posObject = mul(input.posObject,tW);
	output.posScreen = mul(input.posObject,mul(tW,tVP));
	output.uv = mul(input.uv, tTex);
	return output;
}



float4 PS_AT(psInput_AT input): SV_Target
{
    float4 col = 0;
	
	float worldSpaceDistance = distance(lightPos, input.posObject.xyz);
	float dist = (worldSpaceDistance - nearFarPlane.x) /
              (nearFarPlane.y - nearFarPlane.x) + depthOffset;
	
	float alpha = inputTexture.Sample(linearSampler,input.uv.xy).a;

	col.r = saturate(dist);
	col.g = col.r * col.r;
	col.a = pow(alpha,.25);
	
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

technique11 VSM_AlphaTex
{
	pass P1
	{
		SetVertexShader( CompileShader( vs_5_0, VS_AT() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_AT() ) );
	}
}






