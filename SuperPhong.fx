//@author: mburk
//@help: internet
//@tags: shading, blinn
//@credits: Vux, Dottore, Catweasel

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
};
	
	float4x4 NormalTransform <string uiname="Normal Rotation";>;
	float2 KrMin <String uiname="Fresnel Rim/Refl Min ";float uimin=0.0; float uimax=1;> = 0.002 ;
	float2 Kr <String uiname="Fresnel Rim/Refl Max ";float uimin=0.0; float uimax=6.0;> = 0.5 ;
	float2 FresExp <String uiname="Fresnel Rim/Refl Exp ";float ufimin=0.0; float uimax=30;> = 5 ;
	float3 camPos <string uiname="Camera Position";> ;
	float4 RimColor <bool color = true; string uiname="Rim Color";>  = { 0.0f,0.0f,0.0f,0.0f };
	float4 Color <bool color = true; string uiname="Color Overlay";>  = { 1.0f,1.0f,1.0f,1.0f };
	float Alpha <float uimin=0.0; float uimax=1.0;> = 1;
	
	float spotFade <string uiname="SpotLight Fading";> = 1 ;
	float bumpy <string uiname="Bumpiness";> = 1 ;
	float2 reflective <String uiname="Reflective/Diffuse";float2 uimin=0.0; float uimax=1;> = 1 ;
	bool refraction <bool visible=false; String uiname="Refraction";> = false;
	StructuredBuffer <float> refractionIndex <bool visible=false; String uiname="Refraction Index";>;
	bool BPCM <bool visible=false;> = false; String uiname="Box Projected Cube Map";;
	float3 cubeMapPos  <bool visible=false;string uiname="Cube Map Position"; > = float3(0,0,0);
	StructuredBuffer <float3> cubeMapBoxBounds <bool visible=false;string uiname="Cube Map Bounds";>;
	//int reflectMode <bool visible=false;string uiname="ReflectionMode: Mul/Add"; int uimin=0.0; int uimax=1.0;> = 1;
	int diffuseMode <bool visible=false;string uiname="DiffuseAffect: Reflection/Specular/Both"; int uimin=0.0; int uimax=2.0;> = 2;
	
	
	StructuredBuffer <float4x4> texTransforms <string uiname="tColor,tSpec,tDiffuse,tNormal";>;
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
	bool useIridescence = false;
	Texture2D iridescence <string uiname="Iridescence"; >;
	TextureCube cubeTexRefl <string uiname="CubeMap Refl"; >;
	TextureCube cubeTexIrradiance <string uiname="CubeMap Irradiance"; >;
	Texture2DArray lightMap <string uiname="SpotTex"; >;
	Texture2DArray shadowMap <string uiname="ShadowMap"; >;
	StructuredBuffer <int> useShadow <string uiname="Shadow"; >;
	


#include "dx11/PhongPoint.fxh"
#include "dx11/PhongPointSpot.fxh"
#include "dx11/PhongDirectional.fxh"

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


struct vs2ps
{
    float4 PosWVP: SV_POSITION;
    float4 TexCd : TEXCOORD0;
	float4 PosO: TEXCOORD1;
	float4 ViewDirV: TEXCOORD2;
	float4 PosW: TEXCOORD3;
	float4 NormW : TEXCOORD4;
	float4 NormO : TEXCOORD5;

	
//  BumpMap
///////////////////////////////////////
	float4 tangent : TEXCOORD6;
	float4 binormal : TEXCOORD7;
///////////////////////////////////////	

	float4 reflectionPosition : TEXCOORD8;

};

// -----------------------------------------------------------------------------
// VERTEXSHADERS
// -----------------------------------------------------------------------------

vs2ps VS_Bump(
    float4 PosO: POSITION,
    float4 NormO: NORMAL,
    float4 TexCd : TEXCOORD0,
//  BumpMap
///////////////////////////////////////
	float4 tangent : TANGENT,
    float4 binormal : BINORMAL
///////////////////////////////////////
)
{
    //inititalize all fields of output struct with 0
    vs2ps Out = (vs2ps)0;

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
	
	Out.PosO = PosO;
    Out.PosW = mul(PosO, tW);
    Out.NormW = mul(NormO, tWIT);
	Out.NormO = NormO;

//	position (projected)
    Out.PosWVP  = mul(PosO, tWVP);

	Out.TexCd = TexCd;
    Out.ViewDirV = -normalize(mul(PosO, tWV));
	

	
    return Out;
}


static const float minVariance = 0;
float lightBleedingLimit;
float2	nearFarPlane;

float reduceLightBleeding(float p_max, float amount)
{
    return clamp((p_max-amount)/ (1.0-amount), 0.0, 1.0);
}

float4 calcShadowVSM(float worldSpaceDistance, float2 projectTexCoord, int shadowCounter){
	
	    float currentDistanceToLight = clamp((worldSpaceDistance - nearFarPlane.x) 
        / (nearFarPlane.y - nearFarPlane.x), 0, 1);

    /////////////////////////////////////////////////////////

    // get blured and blured squared distance to light

	float2 depths = shadowMap.SampleLevel(shadowSampler, float3(projectTexCoord, shadowCounter), 0 ).xy;

    float M1 = depths.x;
    float M2 = depths.y;
    float M12 = M1 * M1;

    float p = 0.0;
    float lightIntensity = 1;
    if(currentDistanceToLight >= M1)
    {
        // standard deviation
        float sigma2 = M2 - M12;

        // when standard deviation is smaller than epsilon
        if(sigma2 < minVariance)
        {
            sigma2 = minVariance;
        }

        // chebyshev inequality - upper bound on the 
        // probability that fragment is occluded
        float intensity = sigma2 / (sigma2 + pow(currentDistanceToLight - M1, 2));

        // reduce light bleeding
        lightIntensity = reduceLightBleeding(intensity, lightBleedingLimit);
    }

    /////////////////////////////////////////////////////////

    float4 resultingColor = float4(float3(lightIntensity,lightIntensity,lightIntensity),1);
	
	return resultingColor;
	
}

float4 PS_SuperphongBump(vs2ps In): SV_Target
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
	
	uint numTexTrans, dummy;
    texTransforms.GetDimensions(numTexTrans, dummy);
	
//	uint texColCount, tc;
//    texture2d.GetDimensions(texColCount, tc);
	
	float4 texCol = float4(0,0,0,0);
//	if(tc > 0) 
	texCol = texture2d.Sample(g_samLinear, mul(In.TexCd,texTransforms[0%numTexTrans]).xy);

	float4 specIntensity = specTex.Sample(g_samLinear, mul(In.TexCd,texTransforms[1%numTexTrans]).xy);
	float4 diffuse = diffuseTex.Sample(g_samLinear, mul(In.TexCd,texTransforms[2%numTexTrans]).xy);

	{
	if(diffuseMode == 1 || diffuseMode == 2)
		specIntensity *= saturate(length(diffuse.rgb));
	}
	
	float4 newCol = float4(0,0,0,0);
	float4 ambient = float4(0,0,0,0);
	float4 Nn = normalize(In.NormW);
	
//  BumpMap
///////////////////////////////////////
	float4 bumpMap = normalTex.Sample(g_samLinear, mul(In.TexCd,texTransforms[3%numTexTrans]).xy);;
	
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	
    float4 bumpNormal = (bumpMap.x * In.tangent) + (bumpMap.y * In.binormal) + (bumpMap.z * In.NormO);

	
	In.NormO += normalize(bumpNormal)*bumpy;
	
	float3 NormV =  normalize(mul(mul(In.NormO, (float4x4)tWIT),(float4x4)tV).xyz);
	
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
   	float fresRim = KrMin.x + (Kr.x-KrMin.x) * pow(1-abs(vdn),FresExp.x);
	float fresRefl = KrMin.y + (Kr.y-KrMin.y) * pow(1-abs(vdn),FresExp.y);
	float4 reflColor = float4(0,0,0,0);
	float4 reflColorNorm = float4(0,0,0,0);
	float4 refrColor = float4(0,0,0,0);
	
	
		reflColor = cubeTexRefl.Sample(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z));
		reflColorNorm =  cubeTexIrradiance.Sample(g_samLinear,reflVecNorm);
		
		if(refraction){
				float3 refrVect;
			    for(int r=0; r<3; r++) {
			    	refrVect = refract(-Vn, Nb , refractionIndex[r]);
			    	refrColor += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
				}
		}
		
	
		reflColor = lerp(refrColor,reflColor,fresRefl);

		float inverseDotView = 1.0 - max(dot(Nb,Vn),0.0);
		float4 iridescenceColor = float4(0,0,0,0);
		if (useIridescence) iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0))*fresRefl;
		
	
	if(diffuseMode == 0 || diffuseMode ==2){
			reflColor *= saturate(length(diffuse.rgb));
			reflColorNorm *= saturate(length(diffuse.rgb));			
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

	for(uint i = 0; i< numLights; i++){
		
		
		float4 lightToObject = float4(lPos[i],1) - In.PosW;
		float lightDist = length(lightToObject);
		float falloff = lightRange[i%numLighRange]-length(lightToObject);
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
						newCol += PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity);
						newCol *= calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1);					
//						newCol = calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1);	
					} else {
						newCol += PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity);
					}
				} else {
					newCol += PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity);
				}
				ambient += saturate(lAmbient[i%numlAmb]);
				
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
					projectionColor *= saturate(1/(viewPosition.z*spotFade));
					
					if(useShadow[i]){
						projectionColor *= saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));			
					}
					
			  		newCol += PhongPointSpot(lightDist, NormV, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
							  lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lDiff[i%numlDiff],
							  lSpec[i%numlSpec],specIntensity, projectTexCoord,projectionColor,lightRange[i%numLighRange]);
						
				}
			
				ambient += saturate(lAmbient[i%numlAmb]*falloff);
				
				break;
	
			
			case 2:
				
				bool shadowed = false;
			
				lightCounter+=6;
				float4 shadow = 0;
				float pZ;
			
//				LightDirW = normalize(lightToObject);
//				LightDirV = mul(LightDirW, tV);
				
		
				
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
						
							shadow += (calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));

						} 
					}
		
					ambient += (lAmbient[i%numlAmb]*falloff);
		  			newCol += PhongPoint(lightDist, NormV.xyz, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
									 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange]) * saturate(shadow);
	
			
				} else {
					ambient += (lAmbient[i%numlAmb]*falloff);
		  			newCol += PhongPoint(lightDist, NormV, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
						  			 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange]);

				}	
			
				newCol  /=2;
				ambient /=2;
			break;
			
		}
		
		
		
	}
	
	float4 newRefl = (reflColor+iridescenceColor)*reflective.x*saturate(specIntensity) + RimColor * fresRim;
	float4 finalDiffuse = saturate(saturate(newCol) + saturate(ambient) + reflColorNorm * reflective.y);

	newCol += newRefl;
	if(refraction) fresRefl = 1;
	newCol += finalDiffuse*(1 - reflective.x * fresRefl );	


    return saturate(newCol) * Color.rgba  * texCol;
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
	
	uint numTexTrans, dummy;
    texTransforms.GetDimensions(numTexTrans, dummy);
	
	uint texColCount, tc;
    texture2d.GetDimensions(texColCount, tc);
	
	float4 texCol = texture2d.Sample(g_samLinear, mul(In.TexCd,texTransforms[0%numTexTrans]).xy);
	
	float4 specIntensity = specTex.Sample(g_samLinear, mul(In.TexCd,texTransforms[1%numTexTrans]).xy);
	float4 diffuse = diffuseTex.Sample(g_samLinear, mul(In.TexCd,texTransforms[2%numTexTrans]).xy);
	
	float3 NormV =  normalize(mul(mul(In.NormO.xyz, (float3x3)tWIT),(float3x3)tV).xyz);
	
	if(diffuseMode == 1 || diffuseMode == 2){
		specIntensity *= saturate(length(diffuse.rgb));
	}
	
	
	float4 newCol = float4(0,0,0,0);
	float4 ambient = float4(0,0,0,0);

	float4 Nn = normalize(In.NormW);
	
	
// Reflection and RimLight

	float3 Vn = normalize(camPos - In.PosW.xyz);

	float vdn = -saturate(dot(Vn,In.NormW.xyz));

   	float4 fresRim = KrMin.x + (Kr.x-KrMin.x) * (pow(1-abs(vdn),FresExp.x));
	float4 fresRefl = KrMin.y + (Kr.y-KrMin.y) * (pow(1-abs(vdn),FresExp.y));
	
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
	

		
		reflColor = cubeTexRefl.Sample(g_samLinear,float3(reflVect.x, reflVect.y, reflVect.z));
		reflColorNorm =  cubeTexIrradiance.Sample(g_samLinear,reflVecNorm);
		if(refraction){
				float3 refrVect;
			    for(int r=0; r<3; r++) {
			    	refrVect = refract(-Vn, Nn.xyz , refractionIndex[r]);
			    	refrColor += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
				}
		}
		reflColor = lerp(refrColor,reflColor,fresRefl);
		
	
		float inverseDotView = 1.0 - max(dot(Nn.xyz,Vn),0.0);
		float4 iridescenceColor = float4(0,0,0,0);
		if (useIridescence) iridescenceColor = iridescence.Sample(g_samLinear, float2(inverseDotView,0))*fresRefl;

	
	
	if(diffuseMode == 0 || diffuseMode == 2){
			reflColor *= saturate(length(diffuse.rgb));
			reflColorNorm *= saturate(length(diffuse.rgb));	
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

	for(uint i = 0; i< numLights; i++){
		
		
		float4 lightToObject = float4(lPos[i],1) - In.PosW;
		float lightDist = length(lightToObject);
		float falloff = lightRange[i%numLighRange]-length(lightToObject);
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
						newCol += PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity);
						newCol *= calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1);					
//						newCol = calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1);	
					} else {
						newCol += PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity);
					}
				} else {
					newCol += PhongDirectional(NormV, In.ViewDirV.xyz, LightDirV.xyz, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity);
				}
				ambient += saturate(lAmbient[i%numlAmb]);
				
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
					projectionColor *= saturate(1/(viewPosition.z*spotFade));
					
					if(useShadow[i]){
						projectionColor *= saturate(calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1));			
					}
					
			  		newCol += PhongPointSpot(lightDist, NormV, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
							  lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lDiff[i%numlDiff],
							  lSpec[i%numlSpec],specIntensity, projectTexCoord,projectionColor,lightRange[i%numLighRange]);
						
				}
			
				ambient += saturate(lAmbient[i%numlAmb]*falloff);
				
				break;
	
			
			case 2:
				
				bool shadowed = false;
			
				lightCounter+=6;
				float4 shadow = 0;
				float pZ;
			
//				LightDirW = normalize(lightToObject);
//				LightDirV = mul(LightDirW, tV);
				
		
				
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
						
							shadow += (calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));

						} 
					}
		
					ambient += (lAmbient[i%numlAmb]*falloff);
		  			newCol += PhongPoint(lightDist, NormV.xyz, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
									 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange]) * saturate(shadow);
	
			
				} else {
					ambient += (lAmbient[i%numlAmb]*falloff);
		  			newCol += PhongPoint(lightDist, NormV, In.ViewDirV.xyz, LightDirV.xyz, lPos[i],
						  			 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange]);

				}	
			
				newCol  /=2;
				ambient /=2;
			break;
			
		}
		
		
		
	}
	
	float4 newRefl = (reflColor+iridescenceColor)*reflective.x*saturate(specIntensity) + RimColor * fresRim;
	float4 finalDiffuse = saturate(saturate(newCol) + saturate(ambient) + reflColorNorm * reflective.y);

	newCol += newRefl;
	if(refraction) fresRefl = 1;
	newCol += finalDiffuse*(1 - reflective.x * fresRefl );	

	
    return newCol * Color.rgba * texCol;

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
