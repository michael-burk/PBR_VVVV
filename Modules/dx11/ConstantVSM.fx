//@author: vux
//@help: standard constant shader
//@tags: color
//@credits: 

StructuredBuffer <float3> lPos;

struct vsInput
{
    float4 posObject : POSITION;
};

struct psInput
{
    float4 posScreen : SV_Position;
	float4 posView: VIEW;
};



Texture2D inputTexture <string uiname="Texture";>;

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
	float Alpha <float uimin=0.0; float uimax=1.0;> = 1; 
	float4 cAmb <bool color=true;String uiname="Color";> = { 1.0f,1.0f,1.0f,1.0f };
	float4x4 tColor <string uiname="Color Transform";>;
};

cbuffer cbTextureData : register(b2)
{
	float4x4 tTex <string uiname="Texture Transform"; bool uvspace=true; >;
};

psInput VS(vsInput input)
{
	psInput output;
	output.posScreen = mul(input.posObject,mul(tW,tVP));
	output.posView = mul(input.posObject,tVP);
	return output;
}
float2 doMoments(float Depth){
	float2 Moments;
	// First moment is the depth itself.  
	Moments.x = Depth;
	// Compute partial derivatives of depth.  
	float dx = ddx(Depth);
	float dy = ddy(Depth);
	// Compute second moment over the pixel extents.  
	Moments.y = Depth*Depth + 0.25*(dx*dx + dy*dy);
	return Moments;
};

float2 PS(psInput input): SV_Target
{
//    float4 col = cAmb;
//	col = mul(col, tColor);
//	col.a *= Alpha;
//	
//	lPos[0]

	 float DistToLight = length(input.posView);
	 return doMoments(DistToLight);

//    return col;
}



technique11 ConstantNoTexture
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS() ) );
		SetPixelShader( CompileShader( ps_4_0, PS() ) );
	}
}





