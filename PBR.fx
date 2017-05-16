//@author: mburk
//@help: internet
//@tags: shading, blinn
//@credits: Vux, Dottore, Catweasel

struct lightStruct
{
	float4 diffuse : COLOR0;
    float4 reflection : COLOR0;
	float4 ambient : COLOR1;
};

cbuffer cbPerRender : register( b0 )
{
	float4x4 tP: PROJECTION;   //projection matrix as set via Renderer
	float4x4 tV: VIEW;         //view matrix as set via Renderer
};
 
cbuffer cbPerObject : register (b1)
{	
	//transforms
	float4x4 tW: WORLD;        //the models world matrix
	float4x4 tWV: WORLDVIEW;
	float4x4 tWVP: WORLDVIEWPROJECTION;
	float4x4 tWIT: WORLDINVERSETRANSPOSE;

	float4x4 NormalTransform <string uiname="Normal Rotation";>;
	float3 camPos <string uiname="Camera Position";> ;
	float4 GlobalReflectionColor <bool color = true; string uiname="Global Reflection Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	float4 GlobalDiffuseColor <bool color = true; string uiname="Global Diffuse Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	
	float4 Color <bool color = true; string uiname="Color(Albedo)";>  = { 1.0f,1.0f,1.0f,1.0f };
	float Alpha <float uimin=0.0; float uimax=1.0;> = 1;
	float lPower <String uiname="Power"; float uimin=0.0;> = 25.0;     //shininess of specular highlight


	bool refraction <bool visible=false; String uiname="Refraction";> = false;
	bool BPCM <bool visible= false; String uiname="Box Projected Cube Map";>;
	float3 cubeMapPos  <bool visible=false;string uiname="Cube Map Position"; > = float3(0,0,0);
	bool useIridescence = false;
		
	static const float minVariance = 0;	
	
	float4x4 tColor <bool uvspace=true;>;
	float4x4 tNormal <bool uvspace=true;>;
	float4x4 tRoughness <bool uvspace=true;>;
	float4x4 tMetallic <bool uvspace=true;>;
	float4x4 tAO <bool uvspace=true;>;
	
	float2 iblIntensity <bool visible=true; String uiname="IBL Intensity";> = float2(1,1);
	
	
	bool noTile = false;
	
	float bumpy <string uiname="Bumpiness"; float uimin=0.0; float uimax=1.0;> = 0 ;
	float metallic <float uimin=0.0; float uimax=1.0;>;
	float roughness <float uimin=0.0; float uimax=1.0;>;
//	float ao;
	float3 F <bool visible=false; String uiname="FresnelF0";> = { 0.04,0.04,0.04 };
};

StructuredBuffer <float3> cubeMapBoxBounds <bool visible=false;string uiname="Cube Map Bounds";>;
StructuredBuffer <float> refractionIndex <bool visible=false; String uiname="Refraction Index";>;

StructuredBuffer <float4x4> LightVP <string uiname="LightViewProjection";>;
StructuredBuffer <float4x4> LightV <string uiname="LightView";>;
StructuredBuffer <float4x4> LightP <string uiname="LightProjection";>;
StructuredBuffer <float> lightRange <string uiname="LightRange";>;
StructuredBuffer <int> lightType <string uiname="Directional/Spot/Point";>;	
StructuredBuffer <float3> lPos <string uiname="lPos";>;

StructuredBuffer <float> lAtt0 <string uiname="lAtt0";>;
StructuredBuffer <float> lAtt1 <string uiname="lAtt1";>;
StructuredBuffer <float> lAtt2 <string uiname="lAtt2";>;

StructuredBuffer <float4> lAmbient <string uiname="Ambient Color";>;
StructuredBuffer <float4> lDiff <string uiname="Diffuse Color";>;

Texture2D texture2d <string uiname="Texture"; >;
Texture2D normalTex <string uiname="NormalMap"; >;
Texture2D roughTex <string uiname="RoughnessMap"; >;
Texture2D metallTex <string uiname="MetallicMap"; >;
Texture2D aoTex <string uiname="AOMap"; >;
Texture2D brdfLUT <string uiname="brdfLUT"; >;

Texture2D iridescence <string uiname="Iridescence"; >;
TextureCube cubeTexRefl <string uiname="CubeMap Refl"; >;
TextureCube cubeTexIrradiance <string uiname="CubeMap Irradiance"; >;
Texture2DArray lightMap <string uiname="SpotTex"; >;
Texture2DArray shadowMap <string uiname="ShadowMap"; >;
StructuredBuffer <float2> nearFarPlane <string uiname="Shadow Near Plane / Far Plane"; >;
StructuredBuffer <float> lightBleedingLimit <string uiname="Light Bleeding Limit";>;
StructuredBuffer <int> useShadow <string uiname="Shadow"; >;




SamplerState g_samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState shadowSampler
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

#include "dx11/PhongPoint.fxh"
#include "dx11/PhongPointSpot.fxh"
#include "dx11/PhongDirectional.fxh"
#include "dx11/VSM.fxh"
#include "dx11/NoTile.fxh"

struct vs2psBump
{
    float4 PosWVP: SV_POSITION;
    float4 TexCd : TEXCOORD0;
	float4 PosW: TEXCOORD1;
	float3 NormW : TEXCOORD2;
	float3 tangent : TEXCOORD3;
	float3 binormal : TEXCOORD4;
};


struct vs2ps
{
    float4 PosWVP: SV_POSITION;
    float4 TexCd : TEXCOORD0;
	float4 ViewDirV: TEXCOORD2;
	float4 PosW: TEXCOORD3;
	float3 NormW : TEXCOORD4;
};

// -----------------------------------------------------------------------------
// VERTEXSHADERS
// -----------------------------------------------------------------------------

vs2psBump VS_Bump(
    float4 PosO: POSITION,
    float3 NormO: NORMAL,
    float4 TexCd : TEXCOORD0,
	float3 tangent : TANGENT,
    float3 binormal : BINORMAL
)
{
    //inititalize all fields of output struct with 0
    vs2psBump Out = (vs2psBump)0;

    Out.PosW = mul(PosO, tW);	
	Out.NormW = mul(NormO, NormalTransform);
	
//  BumpMap
///////////////////////////////////////
	// Calculate the tangent vector against the world matrix only and then normalize the final value.
    Out.tangent = mul(tangent, tW);
    Out.tangent = normalize(Out.tangent);

    // Calculate the binormal vector against the world matrix only and then normalize the final value.
    Out.binormal = mul(binormal, tW);
    Out.binormal = normalize(Out.binormal);
///////////////////////////////////////

//	position (projected)
    Out.PosWVP  = mul(PosO, tWVP);
	Out.TexCd = TexCd;
	
	
    return Out;
}

vs2ps VS(
    float4 PosO: POSITION,
    float3 NormO: NORMAL,
    float4 TexCd : TEXCOORD0

)
{
    //inititalize all fields of output struct with 0
    vs2ps Out = (vs2ps)0;
	
    Out.PosW = mul(PosO, tW);
	//NormalTransform
	Out.NormW = mul(NormO, NormalTransform);

//	position (projected)
    Out.PosWVP  = mul(PosO, tWVP);

	Out.TexCd = TexCd;
    Out.ViewDirV = -normalize(mul(PosO, tWV));
	

	
    return Out;
}
float2 R : Targetsize;
float4 getTexel( float3 p, Texture2DArray tex )
{
    p.xy = p.xy*R + 0.5;

    float2 i = floor( p.xy);
    float2 f =  p.xy - i;
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
      p.xy.xy = i + f;

     p.xy = ( p.xy - 0.5)/R;
    return tex.SampleLevel(shadowSampler, p, 0);
}
static const float PI = 3.14159265359;

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}  

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(float3(1.0 - roughness,1.0 - roughness,1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}   

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return nom / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}

float3 cookTorrance(float3 V, float3 L, float3 N, float3 albedo, float3 lDiff,
					float3 lAmb, float shadow, float3 projectionColor, float falloff, float lightDist, float lAtt0, float lAtt1, float lAtt2, float3 F0, float attenuation, float roughness, float metallic, float ao){				
    float3 H = normalize(V + L);
    float3 radiance   = lDiff * attenuation * shadow * projectionColor;
    // cook-torrance brdf
    float NDF = DistributionGGX(N, H, roughness);        
    float G   = GeometrySmith(N, V, L, roughness);      
    float3 F  = fresnelSchlick(max(dot(H, V), 0.0), F0);       					        
    float3 kS = F;
    float3 kD = float3(1.0,1.0,1.0) - kS;
    kD *= 1.0 - metallic;	  					        
    float3 nominator    = NDF * G * F;
    float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
    float3 specular   = nominator / denominator;
	specular *= lPower;
    // add to outgoing radiance Lo
    float NdotL = max(dot(N, L), 0.0);                
    float3 returnLight = (kD * albedo.xyz / PI + specular) * radiance * NdotL; 
	return returnLight + lAmb * lAtt0 / pow(lightDist,lAtt2) * falloff * ao;
}

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static const float3x3 ACESInputMat =
{
    {0.59719, 0.35458, 0.04823},
    {0.07600, 0.90834, 0.01566},
    {0.02840, 0.13383, 0.83777}
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static const float3x3 ACESOutputMat =
{
    { 1.60475, -0.53108, -0.07367},
    {-0.10208,  1.10813, -0.00605},
    {-0.00327, -0.07276,  1.07602}
};

float3 RRTAndODTFit(float3 v)
{
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

float3 ACESFitted(float3 color)
{
    color = mul(ACESInputMat, color);

    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    color = mul(ACESOutputMat, color);

    // Clamp to [0, 1]
    color = saturate(color);

    return color;
}

float4 PS_SuperphongBump(vs2psBump In): SV_Target
{	
	// wavelength colors
	const half4 colors[3] =
        {
    	{ 1, 0, 0, 1 },
    	{ 0, 1, 0, 1 },
    	{ 0, 0, 1, 1 },
	};
	
	float3 LightDirW;
	float4 viewPosition;
	float2 projectTexCoord;
	float3 projectionColor;
	float2 reflectTexCoord;
	float4 finalLight = float4(0.0,0.0,0.0,0.0);
	
	uint tX,tY,m;
	float4 texCol = float4(1,1,1,1);
	float texRoughness = 1;
	float aoT = 1;
	float metallicT = 1;
	
	texture2d.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) texCol = texture2d.Sample(g_samLinear, mul(In.TexCd,tColor).xy);
	else if(tX+tY > 2 && noTile) texCol = textureNoTile(texture2d,mul(In.TexCd,tColor).xy);
	
	roughTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) texRoughness = roughTex.Sample(g_samLinear, mul(In.TexCd,tRoughness).xy).r;
	else if(tX+tY > 2 && noTile) texRoughness = textureNoTile(roughTex,mul(In.TexCd,tRoughness).xy).r;
	
	aoTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) aoT = aoTex.Sample(g_samLinear, mul(In.TexCd,tAO).xy).r;
	else if(tX+tY > 2 && noTile) aoT = textureNoTile(aoTex,mul(In.TexCd,tAO).xy).r;
	
	metallTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) metallicT = metallTex.Sample(g_samLinear, mul(In.TexCd,tMetallic).xy).r;
	else if(tX+tY > 2 && noTile) metallicT = textureNoTile(metallTex,mul(In.TexCd,tMetallic).xy).r;

	float3 Nn = normalize(In.NormW);
	
	float4 bumpMap = float4(0,0,0,0);
	
	normalTex.GetDimensions(tX,tY);
	if(tX+tY > 0 && !noTile) bumpMap = normalTex.Sample(g_samLinear, mul(In.TexCd,tNormal).xy);
	else if(tX+tY > 2 && noTile) bumpMap = textureNoTile(normalTex,mul(In.TexCd,tNormal).xy);
	
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	
	float3 Tn = normalize(In.tangent).xyz;
    float3 Bn = normalize(In.binormal.xyz);
	float3 Nb = normalize(Nn.xyz + (-bumpMap.x * Tn + -bumpMap.y * Bn)*bumpy);

	float3 V = normalize(camPos - In.PosW.xyz);
	

	float3 reflVect = -reflect(V,Nb);
	float3 reflVecNorm = Nn.xyz-reflect(Nn.xyz,Nb);
	float3 refrVect = refract(-V, Nb , refractionIndex[0]);



	// Box Projected CubeMap
	////////////////////////////////////////////////////
	
	if(BPCM){
		
		float3 rbmax = (cubeMapBoxBounds[0] - (In.PosW.xyz))/reflVect;
		float3 rbmin = (cubeMapBoxBounds[1] - (In.PosW.xyz))/reflVect;
		
		
		float3 rbminmax = (reflVect>0.0f)?rbmax:rbmin;
		
		float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
		
		float3 posonbox = In.PosW.xyz + reflVect*fa;
		reflVect = posonbox - cubeMapPos;
		
				
		
		rbmax = (cubeMapBoxBounds[0] - (In.PosW.xyz))/reflVecNorm;
		rbmin = (cubeMapBoxBounds[1] - (In.PosW.xyz))/reflVecNorm;
		
		rbminmax = (reflVecNorm>0.0f)?rbmax:rbmin;
		
		fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
		
		posonbox = In.PosW.xyz + reflVecNorm*fa;
		reflVecNorm = posonbox - cubeMapPos;
			
		
		if(refraction){
			rbmax = (cubeMapBoxBounds[0] - (In.PosW.xyz))/refrVect;
			rbmin = (cubeMapBoxBounds[1] - (In.PosW.xyz))/refrVect;
			

			rbminmax = (refrVect>0.0f)?rbmax:rbmin;
			
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			
			posonbox = In.PosW.xyz + refrVect*fa;
			refrVect = posonbox - cubeMapPos;
		}
		
	}
	
	////////////////////////////////////////////////////
	

	float3 reflColor = float3(0,0,0);
	float3 IBL = float3(0,0,0);

	float4 albedo = texCol * saturate(Color) * aoT;
//	aoT *= ao;
	metallicT *= metallic;
	
    float3 F0 = lerp(F, albedo.xyz, metallicT);
	texRoughness *= roughness;
	
	uint tX1,tY1,m2;
	
	cubeTexRefl.GetDimensions(tX,tY);
	cubeTexIrradiance.GetDimensions(tX1,tY1);
	
	if(tX+tY > 2 || tX1+tY1 > 2){
		
		reflColor = cubeTexRefl.SampleLevel(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z),0).rgb;
		
		float3 kS = fresnelSchlickRoughness(max(dot(Nb, V), 0.0), F,texRoughness);
		float3 kD = 1.0 - kS;
		kD *= 1.0 - metallicT;
		IBL = cubeTexIrradiance.Sample(g_samLinear,reflVecNorm).rgb;
		IBL  = IBL * albedo.xyz;
	
		const float MAX_REFLECTION_LOD = 9.0;
		float3 refl = cubeTexRefl.SampleLevel(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z),texRoughness*MAX_REFLECTION_LOD).rgb;
		float2 envBRDF  = brdfLUT.Sample(g_samLinear, float2(max(dot(Nb, V), 0.0), texRoughness)).rb;
		refl = refl * (F * envBRDF.x + envBRDF.y);
		
		IBL  = (kD * IBL *iblIntensity.x + refl*iblIntensity.y) * aoT;
	} 
	
	float4 iridescenceColor = float4(0,0,0,0);
	if (useIridescence){
		float inverseDotView = 1.0 - max(dot(Nb,V),0.0);
		iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0))*1;
	} 
	

	uint d,textureCount;lightMap.GetDimensions(d,d,textureCount);uint dP,textureCountDepth;
	shadowMap.GetDimensions(dP,dP,textureCountDepth); uint numSpotRange, dummySpot; lightRange.GetDimensions(numSpotRange, dummySpot);
	uint numlAmb, dummyAmb;lAmbient.GetDimensions(numlAmb, dummyAmb);uint numlDiff, dummyDiff;lDiff.GetDimensions(numlDiff, dummyDiff);
	uint numlAtt0, dummylAtt0;lAtt0.GetDimensions(numlAtt0, dummylAtt0);
	uint numlAtt1, dummylAtt1;lAtt1.GetDimensions(numlAtt1, dummylAtt1);uint numlAtt2, dummylAtt2;lAtt2.GetDimensions(numlAtt2, dummylAtt2);
	uint numLVP, dummyLVP;LightVP.GetDimensions(numLVP, dummyLVP);uint numLights,lightCount;lightType.GetDimensions(numLights,lightCount);
	uint numLighRange,lightRangeCount;lightRange.GetDimensions(numLighRange,lightRangeCount);
	
	int pL = 0;
	int shadowCounter = 0;
	int lightCounter = 0;
	float4 shadow = 0;
	texRoughness += .05;

	for(uint i = 0; i< numLights; i++){
		
		float3 lightToObject = float4(lPos[i],1) - In.PosW.xyz;
		float3 L = normalize(float4(lPos[i],1) - In.PosW.xyz);
		float lightDist = length(lightToObject);
		float falloff = pow(saturate(lightRange[i%numLighRange]-lightDist),1.5);
		float projectTexCoordZ;
		
		LightDirW = normalize(lightToObject);

			
		switch (lightType[i]){
			
			
			//DIRECTIONAL
			case 0:
				lightCounter ++;
				
				if(useShadow[i]  == 1){
					shadowCounter++;
				} 

				viewPosition = mul(In.PosW, LightVP[i]);
					
				projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   		projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;			
				projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
			
				if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
				&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
					shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));	
				} else {
					shadow = 1;
				}
						if(useShadow[i]){
							finalLight.xyz += cookTorrance(V, L.xyz, Nb.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  lerp(1.0,saturate(shadow),falloff).x, 1.0, 1, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, 1.0, texRoughness, metallicT, aoT);
					} else {
					       	finalLight.xyz += cookTorrance(V, L.xyz, Nb.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  1.0, 1.0, 1, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, 1.0, texRoughness, metallicT, aoT);
					}
				break;
			
			//SPOT
			case 1:
				
				lightCounter ++;
				
				if(useShadow[i]  == 1){
					shadowCounter++;
				} 

				viewPosition = mul(In.PosW, LightVP[i]);
					
				projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   		projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;			
				projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
				
				if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
				&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
					projectionColor = lightMap.Sample(g_samLinear, float3(projectTexCoord, i), 0 ).rgb;
					shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));	
				}
			
				if(useShadow[i]){
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L.xyz, Nb.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						lerp(1.0,saturate(shadow),falloff).x, projectionColor*falloff, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				} else {
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L.xyz, Nb.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						1.0, projectionColor*falloff, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				}
	
				break;
	
			//POINT
			case 2:
				
				bool shadowed = false;
				lightCounter+=6;
				shadow = 0;
				float pZ;
				if(useShadow[i]){
					
					shadowCounter+=6;
					for(int p = 0; p < 6; p++){
						
						float4x4 LightPcropp = LightP[p + lightCounter-6];
				
						LightPcropp._m00 = 1;
						LightPcropp._m11 = 1;
						
						float4x4 LightVPNew = mul(LightV[p + lightCounter-6],LightPcropp);
						
						viewPosition = mul(In.PosW, LightVPNew);
						
						projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
			   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
						projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
					
						if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
						&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
							
							viewPosition = mul(In.PosW, LightVP[p + lightCounter-6]);

							projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
				   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
							projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
							
							shadow += saturate(calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));

						}
					}
							float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L.xyz, Nb.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							lerp(1,saturate(shadow),falloff).x, 1.0, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				} else {
						    float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L.xyz, Nb.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							1, 1, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				}				
			break;			
		}	
	}

	finalLight.xyz += GlobalReflectionColor.xyz * fresnelSchlick(max(dot(Nb, V), 0.0), F0);
	finalLight.xyz += GlobalDiffuseColor.xyz * aoT;
	
	
//		if(refraction){
//			float3 refrVect;
//		    for(int r=0; r<3; r++) {
//		    	refrVect = refract(-Vn, Nb , refractionIndex[r]);
//		    	light.diffuse += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
//		    	
//			}
//	}
	
	finalLight.xyz += IBL.xyz;
	
//	Gamma Correction
//	finalLight.xyz = finalLight.xyz / (finalLight.xyz + float3(1.0,1.0,1.0));
//	finalLight.xyz = pow(abs(finalLight.xyz), 1.0/2.2); 
	
	finalLight.rgb = ACESFitted(finalLight.rgb);
	finalLight.a = Alpha;
	return finalLight;
//	return metallicT;
	
	

}



float4 PS_Superphong(vs2ps In): SV_Target
{	
	// wavelength colors
	const half4 colors[3] =
        {
    	{ 1, 0, 0, 1 },
    	{ 0, 1, 0, 1 },
    	{ 0, 0, 1, 1 },
	};
	
	float3 LightDirW;
	float4 viewPosition;
	float2 projectTexCoord;
	float3 projectionColor;
	float2 reflectTexCoord;
    float4 reflectionColor;
		
	
	float4 finalLight = float4(0.0,0.0,0.0,0.0);
	
	float4 texCol = float4(1,1,1,1);
	float texRoughness = 1;
	float aoT = 1;
	float metallicT = 1;
	
	uint tX,tY,m;
	
	texture2d.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) texCol = texture2d.Sample(g_samLinear, mul(In.TexCd,tColor).xy);
	else if(tX+tY > 2 && noTile) texCol = textureNoTile(texture2d,mul(In.TexCd,tColor).xy);
	
	roughTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) texRoughness = roughTex.Sample(g_samLinear, mul(In.TexCd,tRoughness).xy).r;
	else if(tX+tY > 2 && noTile) texRoughness = textureNoTile(roughTex,mul(In.TexCd,tRoughness).xy).r;
	
	metallTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) metallicT = metallTex.Sample(g_samLinear, mul(In.TexCd,tMetallic).xy);
	else if(tX+tY > 2 && noTile) metallicT = textureNoTile(metallTex,mul(In.TexCd,tMetallic).xy);

	
	aoTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) aoT = aoTex.Sample(g_samLinear, mul(In.TexCd,tAO).xy).r;
	else if(tX+tY > 2 && noTile) aoT = textureNoTile(aoTex,mul(In.TexCd,tAO).xy).r;
	
//	float3 NormV =  normalize(mul(mul(In.Norm.xyz, (float3x3)tWIT),(float3x3)tV).xyz);
	float3 Nn = normalize(In.NormW);
	
	
// Reflection and RimLight

	float3 V = normalize(camPos - In.PosW.xyz);

	float vdn = -saturate(dot(V,In.NormW));

//	float4 fresRefl = KrMin + (Kr-KrMin) * (pow(1-abs(vdn),FresExp));
	float3 reflVect = -reflect(V,Nn.xyz);
	float3 reflVecNorm = Nn.xyz;
	float3 refrVect = refract(-V, Nn.xyz , refractionIndex[0]);
	
	
// Box Projected CubeMap
	////////////////////////////////////////////////////
	
	if(BPCM){
		
		
		float3 rbmax = (cubeMapBoxBounds[0] - (In.PosW.xyz))/reflVect;
		float3 rbmin = (cubeMapBoxBounds[1] - (In.PosW.xyz))/reflVect;
		
		float3 rbminmax = (reflVect>0.0f)?rbmax:rbmin;
		
		float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
		
		float3 posonbox = In.PosW.xyz + reflVect*fa;
		reflVect = posonbox - cubeMapPos;
		
				
		
			rbmax = (cubeMapBoxBounds[0] - (In.PosW.xyz))/reflVecNorm;
			rbmin = (cubeMapBoxBounds[1] - (In.PosW.xyz))/reflVecNorm;
			
			rbminmax = (reflVecNorm>0.0f)?rbmax:rbmin;
			
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			
			posonbox = In.PosW.xyz + reflVecNorm*fa;
			reflVecNorm = posonbox - cubeMapPos;
			
		
		if(refraction){
			rbmax = (cubeMapBoxBounds[0] - (In.PosW.xyz))/refrVect;
			rbmin = (cubeMapBoxBounds[1] - (In.PosW.xyz))/refrVect;
			
			rbminmax = (refrVect>0.0f)?rbmax:rbmin;
			
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			
			posonbox = In.PosW.xyz + refrVect*fa;
			refrVect = posonbox - cubeMapPos;
		}
		
	}
	
////////////////////////////////////////////////////
	
	float3 reflColor = float3(0,0,0);
//	float3 reflColorNorm = float3(0,0,0);
	float3 refrColor = float3(0,0,0);
	
	float3 IBL = float3(0,0,0);
	
//	aoT *= ao;
	metallicT *= metallic;
	float4 albedo =  texCol * saturate(Color) * aoT;
    float3 F0 = lerp(F, albedo.xyz, metallicT);
	texRoughness *=roughness;
	
	
	uint tX1,tY1,m2;
	
	cubeTexRefl.GetDimensions(tX,tY);
	cubeTexIrradiance.GetDimensions(tX1,tY1);
	
	if(tX+tY > 2 || tX1+tY1 > 2){
		reflColor = cubeTexRefl.SampleLevel(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z),0).rgb;
		
		float3 kS = fresnelSchlickRoughness(max(dot(Nn, V), 0.0), F,texRoughness);
		float3 kD = 1.0 - kS;
		kD *= 1.0 - metallicT;
		IBL = cubeTexIrradiance.Sample(g_samLinear,reflVecNorm).rgb;
		IBL  = IBL * albedo.xyz;
	
	const float MAX_REFLECTION_LOD = 9.0;
		float3 refl = cubeTexRefl.SampleLevel(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z),texRoughness*MAX_REFLECTION_LOD).rgb;
		float2 envBRDF  = brdfLUT.Sample(g_samLinear, float2(max(dot(Nn, V), 0.0), texRoughness)).rb;
		refl = refl * (F * envBRDF.x + envBRDF.y);
			
		IBL  = (kD * IBL *iblIntensity.x + refl*iblIntensity.y) * aoT;
	} 
	
	float inverseDotView = 1.0 - max(dot(Nn.xyz,V),0.0);
	float4 iridescenceColor = float4(0,0,0,0);
	if (useIridescence) iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0))*1;


	uint d,textureCount;lightMap.GetDimensions(d,d,textureCount);uint dP,textureCountDepth;
	shadowMap.GetDimensions(dP,dP,textureCountDepth); uint numSpotRange, dummySpot; lightRange.GetDimensions(numSpotRange, dummySpot);
	uint numlAmb, dummyAmb;lAmbient.GetDimensions(numlAmb, dummyAmb);uint numlDiff, dummyDiff;lDiff.GetDimensions(numlDiff, dummyDiff);
	uint numlAtt0, dummylAtt0;lAtt0.GetDimensions(numlAtt0, dummylAtt0);
	uint numlAtt1, dummylAtt1;lAtt1.GetDimensions(numlAtt1, dummylAtt1);uint numlAtt2, dummylAtt2;lAtt2.GetDimensions(numlAtt2, dummylAtt2);
	uint numLVP, dummyLVP;LightVP.GetDimensions(numLVP, dummyLVP);uint numLights,lightCount;lightType.GetDimensions(numLights,lightCount);
	uint numLighRange,lightRangeCount;lightRange.GetDimensions(numLighRange,lightRangeCount);
	
	int pL = 0;
	int shadowCounter = 0;
	int lightCounter = 0;
	float4 shadow = 0;
	texRoughness += .05;
	
	for(uint i = 0; i< numLights; i++){
		
		float3 lightToObject = float4(lPos[i],1) - In.PosW.xyz;
		float3 L = normalize(float4(lPos[i],1) - In.PosW.xyz);
		float lightDist = length(lightToObject);
		float falloff = pow(saturate(lightRange[i%numLighRange]-lightDist),1.5);
//		float falloff = pow(saturate(1 - pow((lightDist/lightRange[i%numLighRange]),4)),2/pow(lightDist,2)+1 );
		float projectTexCoordZ;
		
		LightDirW = normalize(lightToObject);

			
		switch (lightType[i]){
			
			
			//DIRECTIONAL
			case 0:
				lightCounter ++;
				
				if(useShadow[i]  == 1){
					shadowCounter++;
				} 

				viewPosition = mul(In.PosW, LightVP[i]);
					
				projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   		projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;			
				projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
			
				if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
				&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
					shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));	
				} else {
					shadow = 1;
				}
						if(useShadow[i]){
							finalLight.xyz += cookTorrance(V, L.xyz, Nn.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  lerp(1.0,saturate(shadow),falloff).x, 1.0, 1, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, 1.0, texRoughness, metallicT, aoT);
					} else {
					       	finalLight.xyz += cookTorrance(V, L.xyz, Nn.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  1.0, 1.0, 1, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, 1.0, texRoughness, metallicT, aoT);
					}
				break;
			
			//SPOT
			case 1:
				
				lightCounter ++;
				
				if(useShadow[i]  == 1){
					shadowCounter++;
				} 

				viewPosition = mul(In.PosW, LightVP[i]);
					
				projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   		projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;			
				projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
				
				if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
				&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
					projectionColor = lightMap.Sample(g_samLinear, float3(projectTexCoord, i), 0 ).rgb;
					shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));	
				}
			
				if(useShadow[i]){
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L.xyz, Nn.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						lerp(1.0,saturate(shadow),falloff).x, projectionColor*falloff, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				} else {
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L.xyz, Nn.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						1.0, projectionColor*falloff, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				}
	
				break;
	
			//POINT
			case 2:
				
				bool shadowed = false;
				lightCounter+=6;
				shadow = 0;
				float pZ;
				if(useShadow[i]){
					
					shadowCounter+=6;
					for(int p = 0; p < 6; p++){
						
						float4x4 LightPcropp = LightP[p + lightCounter-6];
				
						LightPcropp._m00 = 1;
						LightPcropp._m11 = 1;
						
						float4x4 LightVPNew = mul(LightV[p + lightCounter-6],LightPcropp);
						
						viewPosition = mul(In.PosW, LightVPNew);
						
						projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
			   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
						projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
					
						if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
						&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
							
							viewPosition = mul(In.PosW, LightVP[p + lightCounter-6]);

							projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
				   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
							projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
							
							shadow += saturate(calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));

						}
					}
							float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L.xyz, Nn.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							lerp(1,saturate(shadow),falloff).x, 1.0, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				} else {
						    float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L.xyz, Nn.xyz, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							1, 1, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				}				
			break;			
		}	
	}

	finalLight.xyz += GlobalReflectionColor.xyz * fresnelSchlick(max(dot(Nn, V), 0.0), F0);
	finalLight.xyz += GlobalDiffuseColor.xyz * aoT;
	
	
//		if(refraction){
//			float3 refrVect;
//		    for(int r=0; r<3; r++) {
//		    	refrVect = refract(-Vn, Nb , refractionIndex[r]);
//		    	light.diffuse += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
//		    	
//			}
//	}
	
	
	finalLight.xyz += IBL.xyz;
	
	
	//	Gamma Correction
//	finalLight.xyz = finalLight.xyz / (finalLight.xyz + float3(1.0,1.0,1.0));
//  finalLight.xyz = pow(abs(finalLight.xyz), 1.0/2.2); 
	
	finalLight.rgb = ACESFitted(finalLight.rgb);
	finalLight.a = Alpha;
//	finalLight *= texCol;
	return finalLight;
}


technique10 Superphong
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_Superphong() ) );
	}
}
technique10 Superphong_Bump
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_Bump() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_SuperphongBump() ) );
	}
}
