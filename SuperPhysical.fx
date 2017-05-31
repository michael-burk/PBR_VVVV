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
	float4x4 tWI: WORLDINVERSE;        //the models world matrix
	float4x4 tWVP: WORLDVIEWPROJECTION;
	
	float4 GlobalReflectionColor <bool color = true; string uiname="Global Reflection Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	float4 GlobalDiffuseColor <bool color = true; string uiname="Global Diffuse Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	
	float4 Color <bool color = true; string uiname="Color(Albedo)";>  = { 1.0f,1.0f,1.0f,1.0f };
	float Alpha <float uimin=0.0; float uimax=1.0;> = 1;
	float lPower <String uiname="Power"; float uimin=0.0;> = 1.0;     //shininess of specular highlight

	bool refraction <bool visible=false; String uiname="Refraction";> = false;
	bool BPCM <bool visible= false; String uiname="Box Projected Cube Map";>;
	float3 cubeMapPos  <bool visible=false;string uiname="Cube Map Position"; > = float3(0,0,0);
	bool useIridescence = false;	

	float4x4 tTex <bool uvspace=true;>;
	float4x4 tTexInv <bool uvspace=true;>;
	
	float2 iblIntensity <bool visible=true; String uiname="IBL Intensity";> = float2(1,1);	
	
	float bumpy <string uiname="Bumpiness"; float uimin=0.0; float uimax=1.0;> = 0 ;
	bool pom <string uiname="Parallax Occlusion Mapping";> = false;
	float metallic <float uimin=0.0; float uimax=1.0;>;
	float roughness <float uimin=0.0; float uimax=1.0;>;
	float3 F <bool visible=false; String uiname="FresnelF0";> = { 0.04,0.04,0.04 };
	
	bool noTile = false;
};

cbuffer cbPerRender : register (b1)
{	
	float3 camPos <string uiname="Camera Position";> ;
}

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
Texture2D heightMap <string uiname="HeightMap"; >;
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

#include "dx11/VSM.fxh"
#include "dx11/NoTile.fxh"
#include "dx11/ParallaxOcclusionMapping.fxh"
#include "dx11/ToneMapping.fxh"
#include "dx11/CookTorrance.fxh"


struct vs2ps
{
    float4 PosWVP: SV_POSITION;
    float4 TexCd : TEXCOORD0;
	float3 V: TEXCOORD1;
	float4 PosW: TEXCOORD2;
	float3 NormW : TEXCOORD3;
};

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
	Out.NormW = mul(NormO, (float3x3)transpose(tWI));
	Out.NormW = normalize(Out.NormW);
	// Calculate the tangent vector against the world matrix only and then normalize the final value.
    Out.tangent = mul(tangent, (float3x3)tW);
    Out.tangent = normalize(Out.tangent);
    // Calculate the binormal vector against the world matrix only and then normalize the final value.
    Out.binormal = mul(binormal, (float3x3)tW);
    Out.binormal = normalize(Out.binormal);
    Out.PosWVP  = mul(PosO, tWVP);
	Out.TexCd = mul(TexCd,tTex);
	Out.V = normalize(camPos - Out.PosW.xyz);
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
	Out.NormW = mul(NormO, (float3x3)transpose(tWI));
	Out.NormW = normalize(Out.NormW);
    Out.PosWVP  = mul(PosO, tWVP);
	Out.TexCd = mul(TexCd,tTex);
	Out.V = normalize(camPos - Out.PosW.xyz);	
    return Out;
}

static const half3 wavelength[3] =
{
	{ 1, 0, 0},
	{ 0, 1, 0},
	{ 0, 0, 1},
};

static const float MAX_REFLECTION_LOD = 9.0;

float4 doLighting(float4 PosW, float3 N, float3 V, float4 TexCd){
	
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
	if(tX+tY > 4 && !noTile) texCol = texture2d.Sample(g_samLinear, TexCd.xy);
	else if(tX+tY > 4 && noTile) texCol = textureNoTile(texture2d,TexCd.xy);

	roughTex.GetDimensions(tX,tY);
	if(tX+tY > 4 && !noTile) texRoughness = roughTex.Sample(g_samLinear, TexCd.xy).r;
	else if(tX+tY > 4 && noTile) texRoughness = textureNoTile(roughTex,TexCd.xy).r;
	
	aoTex.GetDimensions(tX,tY);
	if(tX+tY > 4 && !noTile) aoT = aoTex.Sample(g_samLinear, TexCd.xy).r;
	else if(tX+tY > 4 && noTile) aoT = textureNoTile(aoTex,TexCd.xy).r;
	
	metallTex.GetDimensions(tX,tY);
	if(tX+tY > 4 && !noTile) metallicT = metallTex.Sample(g_samLinear, TexCd.xy).r;
	else if(tX+tY > 4 && noTile) metallicT = textureNoTile(metallTex, TexCd.xy).r;

	float3 reflColor = float3(0,0,0);
	float3 IBL = float3(0,0,0);

	float4 albedo = texCol * saturate(Color) * aoT;
	metallicT *= metallic;
	
    float3 F0 = lerp(F, albedo.xyz, metallicT);
	texRoughness *= max(roughness,.02);

	float3 reflVect = -reflect(V,N);
	float3 reflVecNorm = N;

	
	uint tX1,tY1,m1;
	cubeTexRefl.GetDimensions(tX,tY);
	cubeTexIrradiance.GetDimensions(tX1,tY1);
	
	float3 refrColor = 0;
	float3 iridescenceColor = 0;
	float2 envBRDF = 1;
	if (useIridescence){
		float inverseDotView = 1.0 - max(dot(N,V),0.0);
		iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0)).rgb;
	} else {
		iridescenceColor = 1;
	}
	
	float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F,texRoughness);
	float3 kD = 1.0 - kS;
	kD *= 1-metallicT;
	
	if(tX+tY > 4 || tX1+tY1 > 4){
			
		IBL = cubeTexIrradiance.SampleLevel(g_samLinear,reflVecNorm,texRoughness).rgb;
		IBL  = IBL * albedo.xyz;
		
		float3 refl = cubeTexRefl.SampleLevel(g_samLinear,reflVect,texRoughness*MAX_REFLECTION_LOD).rgb;
		envBRDF  = brdfLUT.Sample(g_samLinear, float2(max(dot(N, V), 0.0), texRoughness)).rb;
		
		if(useIridescence){
		  refl = max(refl,iridescenceColor) * (F0 * envBRDF.x + envBRDF.y);
		} else {
		  refl *= (F0 * envBRDF.x + envBRDF.y);
		}

		if(refraction){
			float3 refrVect;
		    for(int r=0; r<3; r++) {
		    	refrVect = refract(-V, N , refractionIndex[r]);
		    	refrColor += cubeTexRefl.SampleLevel(g_samLinear,refrVect,texRoughness*MAX_REFLECTION_LOD).rgb * wavelength[r];
			}
			refrColor = refrColor * (F * envBRDF.x + envBRDF.y);
		
		}
		
		IBL  = ( ((IBL *iblIntensity.x*(kD)) + refrColor*kD) + refl * iblIntensity.y) * aoT;
		
	} else if(useIridescence){
			float2 envBRDF  = brdfLUT.Sample(g_samLinear, float2(max(dot(N, V), 0.0), texRoughness)).rb;
			iridescenceColor *= (F0 * envBRDF.x + envBRDF.y);
			IBL = iridescenceColor / kD;	
	}
	
	envBRDF  = brdfLUT.Sample(g_samLinear, float2(max(dot(N, V), 0.0), texRoughness)).rb;
	IBL += saturate(GlobalReflectionColor.rgb * (F0 * envBRDF.x + envBRDF.y) / kD);
	IBL += GlobalDiffuseColor.rgb * albedo.rgb * kD;
	
	
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
		
		float3 lightToObject = lPos[i] - PosW.xyz;
		float3 L = normalize(lightToObject);
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
											  lerp(1.0,saturate(shadow),falloff).x, 1.0, 1, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, lAtt0[i%numlAtt0], texRoughness, metallicT, aoT,iridescenceColor);
					} else {
					       	finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
											  1.0, 1.0, 1.0, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, lAtt0[i%numlAtt0], texRoughness, metallicT, aoT,iridescenceColor);
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
					shadow = saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));

				}
			
				if(useShadow[i]){
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						lerp(1.0,saturate(shadow),falloff).x, falloffSpot, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT, iridescenceColor);
					
				} else {
						float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
						finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
						1.0, falloffSpot, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT, iridescenceColor);
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
							lerp(1,saturate(shadow),falloff).x, 1.0, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT, iridescenceColor);
				} else {
						    float attenuation = lAtt0[i%numlAtt0] / pow(lightDist,lAtt1[i%numlAtt1]);
							finalLight.xyz += cookTorrance(V, L, N, albedo.xyz, lDiff[i%numlDiff].xyz, lAmbient[i%numlDiff].xyz,
							1, 1, falloff, lightDist, lAtt0[i%numlAtt0], lAtt1[i%numlAtt1], lAtt2[i%numlAtt2], F0, attenuation, texRoughness, metallicT, aoT, iridescenceColor);
				}				
			break;			
		}	
	}
	
	finalLight.xyz += IBL.xyz;
	
//	Gamma Correction

	finalLight.rgb = ACESFitted(finalLight.rgb);
	finalLight.a = Alpha;

	return finalLight;
}


float4 PS_PBR(vs2ps In): SV_Target
{	
	return doLighting(In.PosW, normalize(In.NormW), In.V, In.TexCd);
}

float4 PS_PBR_Bump(vs2psBump In): SV_Target
{	
	
	if(pom){
		parallaxOcclusionMapping(In.TexCd.xy, In.PosW.xyz, In.V, float3x3(In.tangent,In.binormal,In.NormW.xyz));
	}
	
	float3 bumpMap = float3(0,0,0);
	
	uint tX2,tY2,m2;
	normalTex.GetDimensions(tX2,tY2);
	if(tX2+tY2 > 0 && !noTile) bumpMap = normalTex.Sample(g_samLinear, In.TexCd.xy).rgb;
	else if(tX2+tY2 > 2 && noTile) bumpMap = textureNoTile(normalTex, In.TexCd.xy).rgb;
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	float3 Nb = normalize(In.NormW.xyz + (bumpMap.x * In.tangent + bumpMap.y * In.binormal)*bumpy);
	return doLighting(In.PosW, Nb, In.V, In.TexCd);

}

float4 PS_PBR_Bump_AutoTNB(vs2ps In): SV_Target
{	
	
	// compute derivations of the world position
	float3 p_dx = ddx(In.PosW.xyz);
	float3 p_dy = ddy(In.PosW.xyz);
	// compute derivations of the texture coordinate
	float2 tc_dx = ddx(In.TexCd.xy);
	float2 tc_dy = ddy(In.TexCd.xy);
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
	
	if(pom){
		parallaxOcclusionMapping(In.TexCd.xy, In.PosW.xyz, In.V, float3x3(t,b,In.NormW.xyz));
	}
	
	float3 bumpMap = float3(0,0,0);

	uint tX2,tY2,m2;
	normalTex.GetDimensions(tX2,tY2);
	if(tX2+tY2 > 0 && !noTile) bumpMap = normalTex.Sample(g_samLinear,In.TexCd.xy).rgb;
	else if(tX2+tY2 > 2 && noTile) bumpMap = textureNoTile(normalTex,In.TexCd.xy).rgb;
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	
	float3 Nb = normalize(In.NormW.xyz + (bumpMap.x * normalize(t) + bumpMap.y * normalize(b))*bumpy);
	
	return doLighting(In.PosW, Nb, In.V, In.TexCd);
}


technique10 PBR
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_PBR() ) );
	}
}

technique10 PBR_Bump
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_Bump() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_PBR_Bump() ) );
	}
}

technique10 PBR_Bump_AutoTNB
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS() ) );
		SetPixelShader( CompileShader( ps_5_0, PS_PBR_Bump_AutoTNB() ) );
	}
}
