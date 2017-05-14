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
	float KrMin <String uiname="MIN Fresnel";float uimin=0.0; float uimax=1;> = 0 ;
	float Kr <String uiname="MAX Fresnel";float uimin=0.0; float uimax=1.0;> = 1 ;
	float FresExp <String uiname="EXP Fresnel";float uimin=0.0; float uimax=2;> = 1 ;
	float3 camPos <string uiname="Camera Position";> ;
	float4 GlobalReflectionColor <bool color = true; string uiname="Global Reflection Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	float4 GlobalDiffuseColor <bool color = true; string uiname="Global Diffuse Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	
	float4 Color <bool color = true; string uiname="Color(Albedo)";>  = { 1.0f,1.0f,1.0f,1.0f };
	float Alpha <float uimin=0.0; float uimax=1.0;> = 1;
	float lPower <String uiname="Power"; float uimin=0.0;> = 25.0;     //shininess of specular highlight


	float bumpy <string uiname="Bumpiness";> = 0 ;
	bool refraction <bool visible=false; String uiname="Refraction";> = false;
	bool BPCM <bool visible= false; String uiname="Box Projected Cube Map";>;
	float3 cubeMapPos  <bool visible=false;string uiname="Cube Map Position"; > = float3(0,0,0);
	bool useIridescence = false;
		
	static const float minVariance = 0;	
	
	float4x4 tColor;
	float4x4 tSpec;
	float4x4 tDiffuse;
	float4x4 tNormal;
	
	bool noTile = false;
	
	float metallic;
	float roughness;
	float ao;
	float3 F = { 0.04,0.04,0.04 };
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
StructuredBuffer <float4> lSpec <string uiname="Specular Color";>;


Texture2D texture2d <string uiname="Texture"; >;
Texture2D specTex <string uiname="SpecularMap"; >;
Texture2D normalTex <string uiname="NormalMap"; >;
Texture2D diffuseTex <string uiname="DiffuseMap"; >;

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
	float4 PosO: TEXCOORD1;
	float4 ViewDirV: TEXCOORD2;
	float4 PosW: TEXCOORD3;
	float4 NormW : TEXCOORD4;
	float4 NormO : TEXCOORD5;
	float4 tangent : TEXCOORD6;
	float4 binormal : TEXCOORD7;
};


struct vs2ps
{
    float4 PosWVP: SV_POSITION;
    float4 TexCd : TEXCOORD0;
	float4 PosO: TEXCOORD1;
	float4 ViewDirV: TEXCOORD2;
	float4 PosW: TEXCOORD3;
	float4 NormW : TEXCOORD4;
	float4 NormO : TEXCOORD5;
};

// -----------------------------------------------------------------------------
// VERTEXSHADERS
// -----------------------------------------------------------------------------

vs2psBump VS_Bump(
    float4 PosO: POSITION,
    float4 NormO: NORMAL,
    float4 TexCd : TEXCOORD0,
	float4 tangent : TANGENT,
    float4 binormal : BINORMAL
)
{
    //inititalize all fields of output struct with 0
    vs2psBump Out = (vs2psBump)0;

    Out.PosW = mul(PosO, tW);
	Out.PosO = PosO;
	Out.NormO = NormO;
	
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
    Out.ViewDirV = -normalize(mul(PosO, tWV));
	
	
    return Out;
}

vs2ps VS(
    float4 PosO: POSITION,
    float4 NormO: NORMAL,
    float4 TexCd : TEXCOORD0

)
{
    //inititalize all fields of output struct with 0
    vs2ps Out = (vs2ps)0;
	
    Out.PosW = mul(PosO, tW);
	Out.PosO = PosO;
	Out.NormO = NormO;
	
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

float4 PS_SuperphongBump(vs2psBump In): SV_Target
{	
	// wavelength colors
	const half4 colors[3] =
        {
    	{ 1, 0, 0, 1 },
    	{ 0, 1, 0, 1 },
    	{ 0, 0, 1, 1 },
	};
	
	float4 LightDirW;
	float4 LightDirV;
	float4 viewPosition;
	float2 projectTexCoord;
	float4 projectionColor;
	float2 reflectTexCoord;
	
	lightStruct light;
	light.diffuse = float4(0,0,0,0);
	light.reflection = float4(0,0,0,0);
	light.ambient = float4(0,0,0,0);
	
	uint tX,tY,m;

	
	float4 texCol = float4(1,1,1,1);
	float4 specIntensity = float4(1,1,1,1);
	float4 diffuseT = float4(1,1,1,1);
	
	texture2d.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) texCol = texture2d.Sample(g_samLinear, mul(In.TexCd,tColor).xy);
	else if(tX+tY > 2 && noTile) texCol = textureNoTile(texture2d,mul(In.TexCd,tColor).xy);
	
	specTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) specIntensity = specTex.Sample(g_samLinear, mul(In.TexCd,tSpec).xy);
	else if(tX+tY > 2 && noTile) specIntensity = textureNoTile(specTex,mul(In.TexCd,tSpec).xy);
	
	diffuseTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) diffuseT = diffuseTex.Sample(g_samLinear, mul(In.TexCd,tDiffuse).xy);
	else if(tX+tY > 2 && noTile) diffuseT = textureNoTile(diffuseTex,mul(In.TexCd,tDiffuse).xy);

	float4 Nn = normalize(In.NormW);
	
//  BumpMap
///////////////////////////////////////
	float4 bumpMap = float4(0,0,0,0);
	
	normalTex.GetDimensions(tX,tY);
	if(tX+tY > 0 && !noTile) bumpMap = normalTex.Sample(g_samLinear, mul(In.TexCd,tNormal).xy);
	else if(tX+tY > 2 && noTile) bumpMap = textureNoTile(normalTex,mul(In.TexCd,tNormal).xy);
	
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	
    float4 bumpNormal = (bumpMap.x * In.tangent) + (bumpMap.y * In.binormal) + (bumpMap.z * In.NormO);

	In.NormO += normalize(-bumpNormal)*bumpy;
	
	float3 NormV =  normalize(mul(mul(In.NormO.xyz, (float3x3)tWIT),(float3x3)tV).xyz);
   
	float3 Tn = normalize(In.tangent).xyz;
    float3 Bn = normalize(In.binormal.xyz);
	float3 Nb = normalize(Nn.xyz + (bumpMap.x * Tn + bumpMap.y * Bn)*bumpy);
///////////////////////////////////////
	
// Reflection and RimLight
	float3 Vn = normalize(camPos - In.PosW.xyz);
	
//BumpMap
///////////////////////////////////////
	float3 reflVect = -reflect(Vn,Nb);
	float3 reflVecNorm = Nn.xyz-reflect(Nn.xyz,Nb);
	float3 refrVect = refract(-Vn, Nb , refractionIndex[0]);
	
///////////////////////////////////////


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
	
	
	float vdn = -saturate(dot(reflVect,In.NormW.xyz));
	float fresRefl = KrMin + (Kr-KrMin) * pow(1-abs(vdn),FresExp);
	float4 reflColor = float4(0,0,0,0);
	float4 reflColorNorm = float4(0,0,0,0);

	
	cubeTexRefl.GetDimensions(tX,tY);
	if(tX+tY > 0)	
	reflColor = cubeTexRefl.Sample(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z));

	cubeTexIrradiance.GetDimensions(tX,tY);
	if(tX+tY > 0) reflColorNorm =  cubeTexIrradiance.Sample(g_samLinear,reflVecNorm);
	

	
	float4 iridescenceColor = float4(0,0,0,0);
	if (useIridescence){
		float inverseDotView = 1.0 - max(dot(Nb,Vn),0.0);
		iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0))*fresRefl;
	} 
	

	uint d,textureCount;
	lightMap.GetDimensions(d,d,textureCount);
	
	uint dP,textureCountDepth;
	shadowMap.GetDimensions(dP,dP,textureCountDepth);
	
	uint numSpotRange, dummySpot;
    lightRange.GetDimensions(numSpotRange, dummySpot);
	
	uint numlAmb, dummyAmb;
    lAmbient.GetDimensions(numlAmb, dummyAmb);
	uint numlDiff, dummyDiff;
    lDiff.GetDimensions(numlDiff, dummyDiff);
	
	uint numlSpec, dummySpec;
    lSpec.GetDimensions(numlSpec, dummySpec);
	
	uint numlAtt0, dummylAtt0;
    lAtt0.GetDimensions(numlAtt0, dummylAtt0);
	
	uint numlAtt1, dummylAtt1;
    lAtt1.GetDimensions(numlAtt1, dummylAtt1);
	
	uint numlAtt2, dummylAtt2;
    lAtt2.GetDimensions(numlAtt2, dummylAtt2);
	
	uint numLVP, dummyLVP;
    LightVP.GetDimensions(numLVP, dummyLVP);
	
	uint numLights,lightCount;
	lightType.GetDimensions(numLights,lightCount);
	
	uint numLighRange,lightRangeCount;
	lightRange.GetDimensions(numLighRange,lightRangeCount);
	
	int pL = 0;
	int shadowCounter = 0;
	int lightCounter = 0;
	float4 shadow = 0;
	
	for(uint i = 0; i< numLights; i++){
		
		
		float4 lightToObject = float4(lPos[i],1) - In.PosW;
		float lightDist = length(lightToObject);
		float falloff = saturate(lightRange[i%numLighRange]-length(lightToObject));
		float projectTexCoordZ;
		LightDirW = normalize(lightToObject);
		LightDirV = mul(LightDirW, tV);
		
		
		switch (lightType[i]){
			
			case 0:
			
				lightCounter ++;

				if(useShadow[i] == 1){
					
					shadowCounter++;
				
					viewPosition = mul(In.PosW, LightVP[i]);
					
					projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
					projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
		
					if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
					&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
						
						shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));
						light = PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lAmbient[i%numlDiff], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,saturate(shadow),light,lightRange[i%numLighRange],lightDist);

					} else {
						light = PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lAmbient[i%numlDiff], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,1,light,lightRange[i%numLighRange],lightDist);
					}
				} else {
						light = PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lAmbient[i%numlDiff], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity, 1,light,lightRange[i%numLighRange],lightDist);
				}
			
				break;
	
			
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
					
					projectionColor = lightMap.Sample(g_samLinear, float3(projectTexCoord, i), 0 );
					if(useShadow[i]){
						shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));			
					
			  			light = PhongPointSpot(lightDist, NormV, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
							  lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lAmbient[i%numlDiff], lDiff[i%numlDiff],
							  lSpec[i%numlSpec],specIntensity, projectTexCoord,projectionColor,lightRange[i%numLighRange],saturate(shadow),light);
					} else {
						light = PhongPointSpot(lightDist, NormV, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
							  lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lAmbient[i%numlDiff], lDiff[i%numlDiff],
							  lSpec[i%numlSpec],specIntensity, projectTexCoord,projectionColor,lightRange[i%numLighRange],1,light);
					}
					
					
				}

				
				break;
	
			
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
		  			light = PhongPoint(lightDist, NormV.xyz, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
									 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lAmbient[i%numlAmb],lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange],saturate(shadow),light);							
				} else {

		  			light = PhongPoint(lightDist, NormV.xyz, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
									 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lAmbient[i%numlAmb], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange],1,light);
				}	
			
			break;
			
		}
		

		
	}
	
	float4 material = texCol * saturate(Color);
	light.reflection = saturate( saturate(light.reflection) + saturate(reflColor) + saturate(iridescenceColor) + saturate(GlobalReflectionColor) ); 
	light.diffuse = saturate(saturate(light.diffuse) +  saturate(light.ambient) +  saturate(reflColorNorm) + saturate(GlobalDiffuseColor)) * material; 
	
	if(refraction){
			float3 refrVect;
		    for(int r=0; r<3; r++) {
		    	refrVect = refract(-Vn, Nb , refractionIndex[r]);
		    	light.diffuse += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
		    	
			}
	}
	
	light.diffuse = lerp(light.diffuse,max(saturate(light.ambient)*material,saturate(light.reflection)),fresRefl*specIntensity);
	light.diffuse.a *= Alpha;
	

	return light.diffuse;
	
	
	
	

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
	
	float4 LightDirW;
	float4 LightDirV;
	float4 viewPosition;
	float2 projectTexCoord;
	float4 projectionColor;
	float2 reflectTexCoord;
    float4 reflectionColor;
	
		
	lightStruct light;
	light.diffuse = float4(0,0,0,0);
	light.reflection = float4(0,0,0,0);
	light.ambient = float4(0,0,0,0);
	
	float4 finalLight = float4(0.0,0.0,0.0,0.0);
//	float4 reflectance;
//	float4
	
	float4 texCol = float4(1,1,1,1);
	float4 specIntensity = float4(1,1,1,1);
	float4 diffuseT = float4(1,1,1,1);
	
	uint tX,tY,m;
	
	texture2d.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) texCol = texture2d.Sample(g_samLinear, mul(In.TexCd,tColor).xy);
	else if(tX+tY > 2 && noTile) texCol = textureNoTile(texture2d,mul(In.TexCd,tColor).xy);
	
	specTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) specIntensity = specTex.Sample(g_samLinear, mul(In.TexCd,tSpec).xy);
	else if(tX+tY > 2 && noTile) specIntensity = textureNoTile(specTex,mul(In.TexCd,tSpec).xy);
	
	diffuseTex.GetDimensions(tX,tY);
	if(tX+tY > 2 && !noTile) diffuseT = diffuseTex.Sample(g_samLinear, mul(In.TexCd,tDiffuse).xy);
	else if(tX+tY > 2 && noTile) diffuseT = textureNoTile(diffuseTex,mul(In.TexCd,tDiffuse).xy);
	
	float3 NormV =  normalize(mul(mul(In.NormO.xyz, (float3x3)tWIT),(float3x3)tV).xyz);
	float3 Nn = normalize(In.NormW.xyz);
	
	
// Reflection and RimLight

	float3 Vn = normalize(camPos - In.PosW.xyz);

	float vdn = -saturate(dot(Vn,In.NormW.xyz));

	float4 fresRefl = KrMin + (Kr-KrMin) * (pow(1-abs(vdn),FresExp));
	float3 reflVect = -reflect(Vn,Nn.xyz);
	float3 reflVecNorm = Nn.xyz;
	float3 refrVect = refract(-Vn, Nn.xyz , refractionIndex[0]);
	
	
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
	
	float4 reflColor = float4(0,0,0,0);
	float4 reflColorNorm = float4(0,0,0,0);
	float4 refrColor = float4(0,0,0,0);
	
	cubeTexRefl.GetDimensions(tX,tY);
	if(tX+tY > 2) reflColor = cubeTexRefl.Sample(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z));
	
	cubeTexIrradiance.GetDimensions(tX,tY);
	if(tX+tY > 2) reflColorNorm =  cubeTexIrradiance.Sample(g_samLinear,reflVecNorm);
	
	float inverseDotView = 1.0 - max(dot(Nn.xyz,Vn),0.0);
	float4 iridescenceColor = float4(0,0,0,0);
	if (useIridescence) iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0))*fresRefl;


	uint d,textureCount;
	lightMap.GetDimensions(d,d,textureCount);
	
	uint dP,textureCountDepth;
	shadowMap.GetDimensions(dP,dP,textureCountDepth);
	
	uint numSpotRange, dummySpot;
    lightRange.GetDimensions(numSpotRange, dummySpot);
	
	uint numlAmb, dummyAmb;
    lAmbient.GetDimensions(numlAmb, dummyAmb);
	uint numlDiff, dummyDiff;
    lDiff.GetDimensions(numlDiff, dummyDiff);
	
	uint numlSpec, dummySpec;
    lSpec.GetDimensions(numlSpec, dummySpec);
	
	uint numlAtt0, dummylAtt0;
    lAtt0.GetDimensions(numlAtt0, dummylAtt0);
	
	uint numlAtt1, dummylAtt1;
    lAtt1.GetDimensions(numlAtt1, dummylAtt1);
	
	uint numlAtt2, dummylAtt2;
    lAtt2.GetDimensions(numlAtt2, dummylAtt2);
	
	uint numLVP, dummyLVP;
    LightVP.GetDimensions(numLVP, dummyLVP);
	
	uint numLights,lightCount;
	lightType.GetDimensions(numLights,lightCount);
	
	uint numLighRange,lightRangeCount;
	lightRange.GetDimensions(numLighRange,lightRangeCount);
	
	int pL = 0;
	int shadowCounter = 0;
	int lightCounter = 0;
	float4 shadow = 0;
	float4 albedo = texCol * saturate(Color) * ao;
	float3 V = normalize(camPos - In.PosW);
    float3 F0 = lerp(F, albedo, metallic);
	
	for(uint i = 0; i< numLights; i++){
		
		float4 lightToObject = float4(lPos[i],1) - In.PosW;
		float4 L = normalize(float4(lPos[i],1) - In.PosW);
		float lightDist = length(lightToObject);
		float falloff = pow(saturate(lightRange[i%numLighRange]-lightDist),1.5);
		float projectTexCoordZ;
		
		LightDirW = normalize(lightToObject);
		LightDirV = mul(LightDirW, tV);

			
		switch (lightType[i]){
			
			case 0:
			
				lightCounter ++;

				if(useShadow[i] == 1){
					
					shadowCounter++;
				
					viewPosition = mul(In.PosW, LightVP[i]);
					
					projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
					projectTexCoordZ = viewPosition.z / viewPosition.w / 2.0f + 0.5f;
		
					if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)
					&& (saturate(projectTexCoordZ) == projectTexCoordZ)){
						
						shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));
						light = PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lAmbient[i%numlDiff], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,saturate(shadow),light,lightRange[i%numLighRange],lightDist);

					} else {
						light = PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lAmbient[i%numlDiff], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity, 1,light,lightRange[i%numLighRange],lightDist);
					}
				} else {
						light = PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lAmbient[i%numlDiff], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity, 1,light,lightRange[i%numLighRange],lightDist);
				}
			
				break;
	
			
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
					
					
					projectionColor = lightMap.Sample(g_samLinear, float3(projectTexCoord, i), 0 );
					if(useShadow[i]){
						shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));			
					
					  // calculate per-light radiance
					        float3 H = normalize(V + L.xyz);
							float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]) * falloff;
					        float3 radiance   = lDiff[i%numlDiff].xyz * attenuation * saturate(shadow).xyz * projectionColor;
					        // cook-torrance brdf
					        float NDF = DistributionGGX(Nn.xyz, H, roughness);        
					        float G   = GeometrySmith(Nn.xyz, V, L.xyz, roughness);      
					        float3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       					        
					        float3 kS = F;
					        float3 kD = float3(1.0,1.0,1.0) - kS;
					        kD *= 1.0 - metallic;	  					        
					        float3 nominator    = NDF * G * F;
					        float denominator = 4 * max(dot(Nn.xyz, V), 0.0) * max(dot(Nn.xyz, L.xyz), 0.0) + 0.001; 
					        float3 specular   = nominator / denominator;					            
					        // add to outgoing radiance Lo
					        float NdotL = max(dot(Nn.xyz, L.xyz), 0.0);                
					        finalLight.xyz += (kD * albedo.xyz / PI + specular) * radiance * NdotL; 
							finalLight.xyz += lAmbient[i%numlDiff] * lAtt0[i%numlAtt0] / pow(lightDist,lAtt2[i%numlAtt2]) * falloff * ao;
						
					} else {
					        float3 H = normalize(V + L.xyz);
							float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]) * falloff;
					        float3 radiance   = lDiff[i%numlDiff].xyz * attenuation * projectionColor;
					        // cook-torrance brdf
					        float NDF = DistributionGGX(Nn.xyz, H, roughness);        
					        float G   = GeometrySmith(Nn.xyz, V, L.xyz, roughness);      
					        float3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       					        
					        float3 kS = F;
					        float3 kD = float3(1.0,1.0,1.0) - kS;
					        kD *= 1.0 - metallic;	  					        
					        float3 nominator    = NDF * G * F;
					        float denominator = 4 * max(dot(Nn.xyz, V), 0.0) * max(dot(Nn.xyz, L.xyz), 0.0) + 0.001; 
					        float3 specular   = nominator / denominator;					            
					        // add to outgoing radiance Lo
					        float NdotL = max(dot(Nn.xyz, L.xyz), 0.0);                
					        finalLight.xyz += (kD * albedo.xyz / PI + specular) * radiance * NdotL; 
							finalLight.xyz += lAmbient[i%numlDiff] * lAtt0[i%numlAtt0] / pow(lightDist,lAtt2[i%numlAtt2]) * falloff * ao;
					}
					
					
				}
			
				
				break;
	
			
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
					  // calculate per-light radiance
					        float3 H = normalize(V + L.xyz);
							float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]) * falloff;
					        float3 radiance   = lDiff[i%numlDiff].xyz * attenuation * saturate(shadow).xyz;
					        // cook-torrance brdf
					        float NDF = DistributionGGX(Nn.xyz, H, roughness);        
					        float G   = GeometrySmith(Nn.xyz, V, L.xyz, roughness);      
					        float3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       					        
					        float3 kS = F;
					        float3 kD = float3(1.0,1.0,1.0) - kS;
					        kD *= 1.0 - metallic;	  					        
					        float3 nominator    = NDF * G * F;
					        float denominator = 4 * max(dot(Nn.xyz, V), 0.0) * max(dot(Nn.xyz, L.xyz), 0.0) + 0.001; 
					        float3 specular   = nominator / denominator;					            
					        // add to outgoing radiance Lo
					        float NdotL = max(dot(Nn.xyz, L.xyz), 0.0);                
					        finalLight.xyz += (kD * albedo.xyz / PI + specular) * radiance * NdotL; 
							finalLight.xyz += lAmbient[i%numlDiff] * lAtt0[i%numlAtt0] / pow(lightDist,lAtt2[i%numlAtt2]) * falloff * ao;
				} else {
					 	 // calculate per-light radiance
					        float3 H = normalize(V + L.xyz);
						    float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]) * falloff;
					        float3 radiance   = lDiff[i%numlDiff].xyz * attenuation;
					        // cook-torrance brdf
					        float  NDF = DistributionGGX(Nn.xyz, H, roughness);        
					        float  G   = GeometrySmith(Nn.xyz, V, L.xyz, roughness);      
					        float3 F   = fresnelSchlick(max(dot(H, V), 0.0), F0);       					 
							float3 kS = F;
					        float3 kD = float3(1.0,1.0,1.0) - kS;
					        kD *= 1.0 - metallic;	  		
					        float3 nominator    = NDF * G * F;
					        float denominator = 4 * max(dot(Nn.xyz, V), 0.0) * max(dot(Nn.xyz, L.xyz), 0.0) + 0.001; 
					        float3 specular   = nominator / denominator;					       
						    // add to outgoing radiance
					        float NdotL = max(dot(Nn.xyz, L.xyz), 0.0);                
					        finalLight.xyz += (kD * albedo.xyz / PI + specular) * radiance * NdotL; 
							// Ambient Light
							finalLight.xyz += lAmbient[i%numlDiff] * lAtt0[i%numlAtt0] / pow(lightDist,lAtt2[i%numlAtt2]) * falloff * ao;
				}				
			break;			
		}	
	}
	

//	light.reflection = saturate( saturate(light.reflection) + saturate(reflColor) + saturate(iridescenceColor) + saturate(GlobalReflectionColor) ) ; 
//	light.diffuse = saturate(saturate(light.diffuse) +  saturate(light.ambient) +  saturate(reflColorNorm) + saturate(GlobalDiffuseColor)) * material; 
//	
//	if(refraction){
//			float3 refrVect;
//		    for(int r=0; r<3; r++) {
//		    	refrVect = refract(-Vn, Nn.xyz , refractionIndex[r]);
//		    	light.diffuse += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
//		    	
//			}
//	}
//	
//	light.diffuse = lerp(light.diffuse,max(saturate(light.ambient)* material,saturate(light.reflection)),fresRefl*specIntensity);
//	light.diffuse.a *= Alpha;
//	
//	

//	light.diffuse = light.reflection
	

	finalLight.xyz += (GlobalReflectionColor) * fresnelSchlick(max(dot(Nn, V), 0.0), F0);
	finalLight.xyz += (GlobalDiffuseColor) * ao;
	
//	// Gamma Correction
	finalLight.xyz = finalLight.xyz / (finalLight.xyz + float3(1.0,1.0,1.0));
    finalLight.xyz = pow(abs(finalLight.xyz), 1.0/2.2); 
//	

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
