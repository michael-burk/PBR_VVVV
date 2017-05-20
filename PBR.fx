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
static const float minVariance = 0;	

cbuffer cbPerObject : register (b0)
{	
	//transforms
	float4x4 tW: WORLD;        //the models world matrix
	float4x4 tWVP: WORLDVIEWPROJECTION;

	float3 camPos <string uiname="Camera Position";> ;
	float4 GlobalReflectionColor <bool color = true; string uiname="Global Reflection Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	float4 GlobalDiffuseColor <bool color = true; string uiname="Global Diffuse Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	
	float4 Color <bool color = true; string uiname="Color(Albedo)";>  = { 1.0f,1.0f,1.0f,1.0f };
	float Alpha <float uimin=0.0; float uimax=1.0;> = 1;
	float lPower <String uiname="Power"; float uimin=0.0;> = 1.0;     //shininess of specular highlight

	bool refraction <bool visible=false; String uiname="Refraction";> = false;
	bool BPCM <bool visible= false; String uiname="Box Projected Cube Map";>;
	float3 cubeMapPos  <bool visible=false;string uiname="Cube Map Position"; > = float3(0,0,0);
	bool useIridescence = false;	
	
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
	float3 V: TEXCOORD5;
};


struct vs2ps
{
    float4 PosWVP: SV_POSITION;
    float4 TexCd : TEXCOORD0;
	float3 V: TEXCOORD1;
	float4 PosW: TEXCOORD2;
	float3 NormW : TEXCOORD3;
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
//	Out.NormW = mul(NormO, NormalTransform);
	Out.NormW = mul(NormO, tW);
	Out.NormW = normalize(Out.NormW);
	// Calculate the tangent vector against the world matrix only and then normalize the final value.
    Out.tangent = mul(tangent, tW);
    Out.tangent = normalize(Out.tangent);
    // Calculate the binormal vector against the world matrix only and then normalize the final value.
    Out.binormal = mul(binormal, tW);
    Out.binormal = normalize(Out.binormal);
    Out.PosWVP  = mul(PosO, tWVP);
	Out.TexCd = TexCd;
	Out.V = normalize(camPos - Out.PosW);
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
//	Out.NormW = mul(NormO, NormalTransform);
	Out.NormW = mul(NormO, tW);
	Out.NormW = normalize(Out.NormW);
    Out.PosWVP  = mul(PosO, tWVP);
	Out.TexCd = TexCd;
	Out.V = normalize(camPos - Out.PosW);
    //Out.ViewDirV = -normalize(mul(PosO, tW));
	
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
					float3 lAmb, float shadow, float3 projectionColor, float falloff,
					float lightDist, float lAtt0, float lAtt1, float lAtt2, float3 F0,
					float attenuation, float roughness, float metallic, float ao){
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


float4 doLighting(float4 PosW, float3 N, float3 V, float4 TexCd){
	
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
	if(tX+tY > 4 && !noTile) texCol = texture2d.Sample(g_samLinear, mul(TexCd,tColor).xy);
	else if(tX+tY > 4 && noTile) texCol = textureNoTile(texture2d,mul(TexCd,tColor).xy);
	
	roughTex.GetDimensions(tX,tY);
	if(tX+tY > 4 && !noTile) texRoughness = roughTex.Sample(g_samLinear, mul(TexCd,tRoughness).xy).r;
	else if(tX+tY > 4 && noTile) texRoughness = textureNoTile(roughTex,mul(TexCd,tRoughness).xy).r;
	
	aoTex.GetDimensions(tX,tY);
	if(tX+tY > 4 && !noTile) aoT = aoTex.Sample(g_samLinear, mul(TexCd,tAO).xy).r;
	else if(tX+tY > 4 && noTile) aoT = textureNoTile(aoTex,mul(TexCd,tAO).xy).r;
	
	metallTex.GetDimensions(tX,tY);
	if(tX+tY > 4 && !noTile) metallicT = metallTex.Sample(g_samLinear, mul(TexCd,tMetallic).xy).r;
	else if(tX+tY > 4 && noTile) metallicT = textureNoTile(metallTex,mul(TexCd,tMetallic).xy).r;
	

	float3 reflColor = float3(0,0,0);
	float3 IBL = float3(0,0,0);

	float4 albedo = texCol * saturate(Color) * aoT;
	metallicT *= metallic;
	
    float3 F0 = lerp(F, albedo.xyz, metallicT);
	texRoughness *= roughness;

	float3 reflVect = -reflect(V,N);
	float3 reflVecNorm = N;
	float3 refrVect = refract(-V, N , refractionIndex[0]);

	// Box Projected CubeMap
	////////////////////////////////////////////////////
	
	if(BPCM){
		
		float3 rbmax = (cubeMapBoxBounds[0] - (PosW))/reflVect;
		float3 rbmin = (cubeMapBoxBounds[1] - (PosW))/reflVect;	
		float3 rbminmax = (reflVect>0.0f)?rbmax:rbmin;	
		float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);	
		float3 posonbox = PosW + reflVect*fa;
		reflVect = posonbox - cubeMapPos;
		rbmax = (cubeMapBoxBounds[0] - (PosW))/reflVecNorm;
		rbmin = (cubeMapBoxBounds[1] - (PosW))/reflVecNorm;
		rbminmax = (reflVecNorm>0.0f)?rbmax:rbmin;	
		fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);	
		posonbox = PosW + reflVecNorm*fa;
		reflVecNorm = posonbox - cubeMapPos;	
		if(refraction){
			rbmax = (cubeMapBoxBounds[0] - (PosW))/refrVect;
			rbmin = (cubeMapBoxBounds[1] - (PosW))/refrVect;
			rbminmax = (refrVect>0.0f)?rbmax:rbmin;		
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);			
			posonbox = PosW + refrVect*fa;
			refrVect = posonbox - cubeMapPos;
		}		
	}
	
	
	uint tX1,tY1,m1;
	cubeTexRefl.GetDimensions(tX,tY);
	cubeTexIrradiance.GetDimensions(tX1,tY1);
	
	if(tX+tY > 2 || tX1+tY1 > 2){
		reflColor = cubeTexRefl.SampleLevel(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z),0).rgb;
		
		float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F,texRoughness);
		float3 kD = 1.0 - kS;
		kD *= 1.0 - metallicT;
		IBL = cubeTexIrradiance.Sample(g_samLinear,reflVecNorm).rgb;
		IBL  = IBL * albedo.xyz;
	
		const float MAX_REFLECTION_LOD = 9.0;
		float3 refl = cubeTexRefl.SampleLevel(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z),texRoughness*MAX_REFLECTION_LOD).rgb;
		float2 envBRDF  = brdfLUT.Sample(g_samLinear, float2(max(dot(N, V), 0.0), texRoughness)).rb;
		refl = refl * (F * envBRDF.x + envBRDF.y);
		
		IBL  = (kD * IBL *iblIntensity.x + refl*iblIntensity.y) * aoT;
	} 
	
	float4 iridescenceColor = float4(0,0,0,0);
	if (useIridescence){
		float inverseDotView = 1.0 - max(dot(N,V),0.0);
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
		
		float3 lightToObject = float4(lPos[i],1) - PosW;
		float3 L = normalize(float4(lPos[i],1) - PosW);
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

				viewPosition = mul(PosW, LightVP[i]);
					
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
							finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  lerp(1.0,saturate(shadow),falloff).x, 1.0, 1, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, lAtt0[i%numlAtt0], texRoughness, metallicT, aoT);
					} else {
					       	finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  1.0, 1.0, 1.0, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, lAtt0[i%numlAtt0], texRoughness, metallicT, aoT);
					}
				break;
			
			//SPOT
			case 1:
				
				lightCounter ++;

				if(useShadow[i]  == 1){
					shadowCounter++;
				} 

				viewPosition = mul(PosW, LightVP[i]);
					
				projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   		projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;			
				projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
				float falloffSpot = 0;
				if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
				&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
					
					uint tXS,tYS,mS;
					lightMap.GetDimensions(mS,tXS,tYS);
					if(tXS+tYS > 4) falloffSpot = lightMap.Sample(g_samLinear, float3(projectTexCoord, i), 0 ).r;
					else if(tXS+tYS < 4) falloffSpot = lerp(1,0,saturate(length(.5-projectTexCoord.xy)*2));
//					falloffSpot = lerp(falloffSpot,0,saturate(length(.5-projectTexCoord.xy)*2));
					
					shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));
				}
			
				if(useShadow[i]){
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						lerp(1.0,saturate(shadow),falloff).x, falloffSpot, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				} else {
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						1.0, falloffSpot, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
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
						
						viewPosition = mul(PosW, LightVPNew);
						
						projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
			   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
						projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
					
						if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
						&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
							
							viewPosition = mul(PosW, LightVP[p + lightCounter-6]);

							projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
				   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
							projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
							
							shadow += saturate(calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));

						}
					}
							float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							lerp(1,saturate(shadow),falloff).x, 1.0, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				} else {
						    float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							1, 1, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT);
				}				
			break;			
		}	
	}

	finalLight.xyz += GlobalReflectionColor.xyz * fresnelSchlick(max(dot(N, V), 0.0), F0);
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
}

float4 PS_SuperphongBump(vs2psBump In): SV_Target
{	

	float4 bumpMap = float4(0,0,0,0);
	
	uint tX2,tY2,m2;
	normalTex.GetDimensions(tX2,tY2);
	if(tX2+tY2 > 0 && !noTile) bumpMap = normalTex.Sample(g_samLinear, mul(In.TexCd,tNormal).xy);
	else if(tX2+tY2 > 2 && noTile) bumpMap = textureNoTile(normalTex,mul(In.TexCd,tNormal).xy);
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	float3 Nb = normalize(In.NormW.xyz + (bumpMap.x * normalize(In.tangent).xyz + -bumpMap.y * normalize(In.binormal.xyz))*bumpy);
//	float3 V = normalize(camPos - In.PosW.xyz);
	return doLighting(In.PosW, Nb, In.V, In.TexCd);
	

}



float4 PS_Superphong(vs2ps In): SV_Target
{	
	return doLighting(In.PosW, In.NormW, In.V, In.TexCd);
}

float4 PS_SuperphongAutoTNB(vs2ps In): SV_Target
{	
	
// compute derivations of the world position
	float3 p_dx = ddx(In.PosW);
	float3 p_dy = ddy(In.PosW);
	// compute derivations of the texture coordinate
	float2 tc_dx = ddx(In.TexCd);
	float2 tc_dy = ddy(In.TexCd);
	// compute initial tangent and bi-tangent
	float3 t = normalize( tc_dy.y * p_dx - tc_dx.y * p_dy );
	float3 b = normalize( tc_dy.x * p_dx - tc_dx.x * p_dy ); // sign inversion
	// get new tangent from a given mesh normal
	float3 n = normalize(In.NormW);
	float3 x = cross(n, t);
	t = cross(x, n);
	t = normalize(t);
	// get updated bi-tangent
	x = cross(b, n);
	b = cross(n, x);
	b = normalize(b);
	
	float4 bumpMap = float4(0,0,0,0);
	
	uint tX2,tY2,m2;
	normalTex.GetDimensions(tX2,tY2);
	if(tX2+tY2 > 0 && !noTile) bumpMap = normalTex.Sample(g_samLinear, mul(In.TexCd,tNormal).xy);
	else if(tX2+tY2 > 2 && noTile) bumpMap = textureNoTile(normalTex,mul(In.TexCd,tNormal).xy);
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	
	float3 Nb = normalize(In.NormW.xyz + (bumpMap.x * normalize(t) + -bumpMap.y * normalize(b))*bumpy);
	
	return doLighting(In.PosW, Nb, In.V, In.TexCd);
}

technique10 PBR
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_Superphong() ) );
	}
}
technique10 PBR_Bump
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_Bump() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_SuperphongBump() ) );
	}
}

technique10 PBR_Bump_AutoTNB
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_SuperphongAutoTNB() ) );
	}
}
