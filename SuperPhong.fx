//@author: mburk
//@help: internet
//@tags: shading, blinn
//@credits: Vux, Dottore, Catweasel
	float SceneScale = .01;
	float LightSize = 1.5;
 	static	float PCFSamples = 9;	// reduce this for higher performance
 	static	float shadowsearchSamples = 12;   // how many samples to use for blocker search
	float shadowLightness = 0;

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
	
	float shadowMapBias = .2;
	
	StructuredBuffer <float4x4> texTransforms <string uiname="tColor,tSpec,tDiffuse,tNormal";>;
	StructuredBuffer <float4x4> LightVP <string uiname="LightView";>;
	StructuredBuffer <float4x4> LightP <string uiname="LightProjection";>;
	StructuredBuffer <float> lightRange <string uiname="LightRange";>;
	StructuredBuffer <int> lightType <string uiname="Directional/Spot/Point";>;	
	StructuredBuffer <float3> lPos <string uiname="lPos";>;

	StructuredBuffer <float> lAtt0 <string uiname="lAtt0";>;
	StructuredBuffer <float> lAtt1 <string uiname="lAtt1";>;
	StructuredBuffer <float> lAtt2 <string uiname="lAtt2";>;

	
//	float4 lAmb <bool color = true; string uiname="Ambient Colo";>  = { 0.0f,0.0f,0.0f,1.0f };
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
#include "dx11/PCSS.fxh"

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
	float3 ViewDirV: TEXCOORD2;
	float3 PosW: TEXCOORD3;
	float3 NormW : TEXCOORD4;
	float3 NormO : TEXCOORD5;

	
//  BumpMap
///////////////////////////////////////
	float3 tangent : TEXCOORD6;
	float3 binormal : TEXCOORD7;
///////////////////////////////////////	

	float4 reflectionPosition : TEXCOORD8;

};

// -----------------------------------------------------------------------------
// VERTEXSHADERS
// -----------------------------------------------------------------------------

vs2ps VS_Bump(
    float4 PosO: POSITION,
    float3 NormO: NORMAL,
    float4 TexCd : TEXCOORD0,
//  BumpMap
///////////////////////////////////////
	float3 tangent : TANGENT,
    float3 binormal : BINORMAL
///////////////////////////////////////
)
{
    //inititalize all fields of output struct with 0
    vs2ps Out = (vs2ps)0;

    Out.PosW = mul(PosO, tW).xyz;
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
    Out.ViewDirV = -normalize(mul(PosO, tWV).xyz);
	
	
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
	
	Out.PosO = PosO;
    Out.PosW = mul(PosO, tW).xyz;
    Out.NormW = mul(NormO, NormalTransform);
	Out.NormO = NormO;

//	position (projected)
    Out.PosWVP  = mul(PosO, tWVP);

	Out.TexCd = TexCd;
    Out.ViewDirV = -normalize(mul(PosO, tWV).xyz);
	

	
    return Out;
}

// -----------------------------------------------------------------------------
// PIXELSHADERS:
// -----------------------------------------------------------------------------

float calcShadow(float3 seed, float4 viewPosition, float2 projectTexCoord, int shadowCounter){
	////////// NEW ///
					
//					http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/#Poisson_Sampling	
//					float bias = 0.005*tan(acos(cosTheta)); // cosTheta is dot( n,l ), clamped between 0 and 1
//					bias = clamp(bias, 0,0.01);
	
	
	
					   // ---------------------------------------------------------
					   // Step 1: Find blocker estimate
					
					   float zReceiver = viewPosition.z ;
					   float searchWidth = SceneScale * (zReceiver - 1.0) / zReceiver;
					   float blocker = findBlocker(float3(projectTexCoord, shadowCounter-1), viewPosition-float4(0,0,shadowMapBias,0), shadowSampler, shadowMapBias,
					                              SceneScale * LightSize / (viewPosition.z), shadowsearchSamples);
					   
					   //return (blocker*1);  // uncomment to visualize blockers
					   
					   // ---------------------------------------------------------
					   // Step 2: Estimate penumbra using parallel planes approximation
					   float penumbra;  
					   penumbra = estimatePenumbra(viewPosition, blocker, LightSize);
					
					  // return penumbra*32;  // uncomment to visualize penumbrae
					
					   // ---------------------------------------------------------
					   // Step 3: Compute percentage-closer filter
					   // based on penumbra estimate

					
					   // Now do a penumbra-based percentage-closer filter
					   float shadowed; 
					
					   shadowed = PCF_Filter(seed,float3(projectTexCoord, shadowCounter-1), viewPosition-float4(0,0,shadowMapBias,0), shadowSampler, shadowMapBias, penumbra, PCFSamples);
									
					
					/////////////////
					return shadowed;
}


float minVariance;
float lightBleedingLimit;
float2	nearFarPlane;
float depthMultiplier;
float epsilon;

float reduceLightBleeding(float p_max, float amount)
{
    return clamp((p_max-amount)/ (1.0-amount), 0.0, 1.0);
}

float4 calcShadowVSM(float worldSpaceDistance, float2 projectTexCoord, int shadowCounter){
	
	    float currentDistanceToLight = clamp((worldSpaceDistance - nearFarPlane.x) 
        / (nearFarPlane.y - nearFarPlane.x), 0, 1);

    /////////////////////////////////////////////////////////

    // get blured and blured squared distance to light

	float2 depths = shadowMap.Sample(shadowSampler, float3(projectTexCoord, shadowCounter), 0 ).xy;

    float M1 = depths.x;
    float M2 = depths.y;
    float M12 = M1 * M1;

    float p = 0.0;
    float lightIntensity = 1.0;
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

float4 calcShadowESM(float worldSpaceDistance, float2 projectTexCoord, int shadowCounter){
	  /////////////////////////////////////////////////

    // current distance to light
        float currentDistanceToLight = clamp((worldSpaceDistance - nearFarPlane.x) 
        / (nearFarPlane.y - nearFarPlane.x), 0, 1);


    /////////////////////////////////////////////////

    // get blured exp of depth
   // float3 projectedCoords = o_shadowCoord.xyz / o_shadowCoord.w;
//    float depthCExpBlured = texture(u_textureShadowMap, projectedCoords.xy).r;
	float depthCExpBlured = shadowMap.Sample(shadowSampler, float3(projectTexCoord, shadowCounter-1), 0 ).r;
    // current exp of depth
    float depthCExpActual = exp(- (depthMultiplier * currentDistanceToLight));
    float expFactor = depthCExpBlured * depthCExpActual;

    // Threshold classification for high frequency artifacts
    if(expFactor > 1.0 + epsilon)
    {
        expFactor = 1.0;
    }

    /////////////////////////////////////////////////

    float4 resultingColor = float4(expFactor,expFactor,expFactor,1);
    
	return expFactor;
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
	
	float3 LightDirW;
	float3 LightDirV;
	float4 viewPosition;
	float2 projectTexCoord;
	float4 projectionColor;
	float2 reflectTexCoord;
	float4 reflectionColor;
	
	uint numTexTrans, dummy;
    texTransforms.GetDimensions(numTexTrans, dummy);
	
	uint texColCount, tc;
    texture2d.GetDimensions(texColCount, tc);
	
	float4 texCol = float4(0,0,0,0);
	if(tc > 0) texCol = texture2d.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[0%numTexTrans]));
	
	float4 specIntensity = specTex.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[1%numTexTrans]));
	float4 diffuse = diffuseTex.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[2%numTexTrans]));

	{
	if(diffuseMode == 1 || diffuseMode == 2)
		specIntensity *= saturate(length(diffuse.rgb));
	}
	
	float3 newCol = float3(0,0,0);
	float3 ambient = float3(0,0,0);
	float3 Nn = normalize(In.NormW);
	
//  BumpMap
///////////////////////////////////////
	float4 bumpMap = normalTex.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[3%numTexTrans]));;
	
	bumpMap = (bumpMap * 2.0f) - 1.0f;
	
    float3 bumpNormal = (bumpMap.x * In.tangent) + (bumpMap.y * In.binormal) + (bumpMap.z * In.NormO);

	
	In.NormO += normalize(bumpNormal)*bumpy;
	
	float3 NormV =  normalize(mul(mul(In.NormO, (float3x3)tWIT),(float3x3)tV).xyz);
	
    float3 Tn = normalize(In.tangent);
    float3 Bn = normalize(In.binormal);
	float3 Nb = normalize(Nn + (bumpMap.x * Tn + bumpMap.y * Bn)*bumpy);
///////////////////////////////////////
	
// Reflection and RimLight
	float3 Vn = normalize(camPos - In.PosW);
	
//BumpMap
///////////////////////////////////////
	float3 reflVect = -reflect(Vn,Nb);
	float3 reflVecNorm = Nn-reflect(Nn,Nb);
	float3 refrVect = refract(-Vn, Nb , refractionIndex[0]);
	
///////////////////////////////////////

	
	// Box Projected CubeMap
	////////////////////////////////////////////////////
	
	if(BPCM){
		
		
		float3 rbmax = (cubeMapBoxBounds[0] - (In.PosW))/reflVect;
		float3 rbmin = (cubeMapBoxBounds[1] - (In.PosW))/reflVect;
		
		
		float3 rbminmax = (reflVect>0.0f)?rbmax:rbmin;
		
		float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
		
		float3 posonbox = In.PosW + reflVect*fa;
		reflVect = posonbox - cubeMapPos;
		
				
		
		rbmax = (cubeMapBoxBounds[0] - (In.PosW))/reflVecNorm;
		rbmin = (cubeMapBoxBounds[1] - (In.PosW))/reflVecNorm;
		
		rbminmax = (reflVecNorm>0.0f)?rbmax:rbmin;
		
		fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
		
		posonbox = In.PosW + reflVecNorm*fa;
		reflVecNorm = posonbox - cubeMapPos;
			
		
		if(refraction){
			rbmax = (cubeMapBoxBounds[0] - (In.PosW))/refrVect;
			rbmin = (cubeMapBoxBounds[1] - (In.PosW))/refrVect;
			

			rbminmax = (refrVect>0.0f)?rbmax:rbmin;
			
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			
			posonbox = In.PosW + refrVect*fa;
			refrVect = posonbox - cubeMapPos;
		}
		
	}
	
	////////////////////////////////////////////////////
	
	
	float vdn = -saturate(dot(reflVect,In.NormW));
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
	
	
	for(int i = 0; i<= numLights; i++){
		float3 lightToObject = lPos[i] - In.PosW;
		switch (lightType[i]){
			case 0:
				LightDirV = normalize(-mul(lPos[i], tV));
				newCol += PhongDirectional(NormV, In.ViewDirV, LightDirV, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity).rgb;
				break;

			
			case 1:
				
				if(length(lightToObject) < lightRange[i%numSpotRange]){	
					
					LightDirW = normalize(lightToObject);
					LightDirV = mul(float4(LightDirW,0.0f), tV).xyz;
			  		newCol += PhongPoint(In.PosW, NormV, In.ViewDirV, LightDirV, lPos[i], lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,lightRange[i%numSpotRange]).rgb;
					
				}
			
				break;
			
			case 2:
			
//				if(length(lightToObject) < lightRange[i%numSpotRange] && dot(lightToObject,SpotLightDir[i%textureCountDepth]) < 0){
					viewPosition = mul(In.PosO, tW);
					viewPosition = mul(viewPosition, LightVP[i%numLVP]);
					
					projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
					
					float3 coords = float3(projectTexCoord, i % textureCount);	//make sure Instance ID buffer is in floats
					
					float shadowMapDepth = shadowMap.Sample(g_samLinear, coords, 0 );
					if ( shadowMapDepth < viewPosition.z) break;
					
					projectionColor = lightMap.Sample(g_samLinear, coords, 0 );
					
					projectionColor *= saturate(1/(viewPosition.z*spotFade));					
					LightDirW = normalize(lightToObject);
					LightDirV = mul(float4(LightDirW,0.0f), tV).xyz;
			  		newCol += PhongPointSpot(In.PosW, NormV, In.ViewDirV, LightDirV, lPos[i], lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity, projectTexCoord,projectionColor,lightRange[i%numSpotRange]).rgb;
					
//				}
			
				break;
		}
		
		
	}
	
	float3 newRefl = (reflColor+iridescenceColor)*reflective.x*saturate(specIntensity) + RimColor * fresRim;
//	float3 finalDiffuse = (newCol + lAmb.rgb + reflColorNorm * reflective.y) * texCol;
	float3 finalDiffuse = saturate(saturate(newCol) + saturate(ambient) + reflColorNorm * reflective.y);

	newCol += newRefl;
	if(refraction) fresRefl = 1;
	newCol += finalDiffuse*(1 - reflective.x * fresRefl );	


    return saturate(float4(newCol, Alpha)) * Color.rgba  * texCol;
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
	
	float4 texCol = texture2d.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[0%numTexTrans]));
	
	float4 specIntensity = specTex.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[1%numTexTrans]));
	float4 diffuse = diffuseTex.Sample(g_samLinear, mul(In.TexCd.xy,texTransforms[2%numTexTrans]));
	
	float3 NormV =  normalize(mul(mul(In.NormO, (float3x3)tWIT),(float3x3)tV).xyz);
	
	if(diffuseMode == 1 || diffuseMode == 2){
		specIntensity *= saturate(length(diffuse.rgb));
	}
	
	
	float3 newCol = float4(0,0,0,0);
	float3 ambient = float4(0,0,0,0);

	float3 Nn = normalize(In.NormW);
	
	
// Reflection and RimLight

	float3 Vn = normalize(camPos - In.PosW);

	float vdn = -saturate(dot(Vn,In.NormW));

   	float4 fresRim = KrMin.x + (Kr.x-KrMin.x) * (pow(1-abs(vdn),FresExp.x));
	float4 fresRefl = KrMin.y + (Kr.y-KrMin.y) * (pow(1-abs(vdn),FresExp.y));
	
	float3 reflVect = -reflect(Vn,Nn);
	float3 reflVecNorm = Nn;
	float3 refrVect = refract(-Vn, Nn , refractionIndex[0]);
	
	
// Box Projected CubeMap
	////////////////////////////////////////////////////
	
	if(BPCM){
		
		
		float3 rbmax = (cubeMapBoxBounds[0] - (In.PosW))/reflVect;
		float3 rbmin = (cubeMapBoxBounds[1] - (In.PosW))/reflVect;
		
		float3 rbminmax = (reflVect>0.0f)?rbmax:rbmin;
		
		float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
		
		float3 posonbox = In.PosW + reflVect*fa;
		reflVect = posonbox - cubeMapPos;
		
				
		
			rbmax = (cubeMapBoxBounds[0] - (In.PosW))/reflVecNorm;
			rbmin = (cubeMapBoxBounds[1] - (In.PosW))/reflVecNorm;
			
			rbminmax = (reflVecNorm>0.0f)?rbmax:rbmin;
			
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			
			posonbox = In.PosW + reflVecNorm*fa;
			reflVecNorm = posonbox - cubeMapPos;
			
		
		if(refraction){
			rbmax = (cubeMapBoxBounds[0] - (In.PosW))/refrVect;
			rbmin = (cubeMapBoxBounds[1] - (In.PosW))/refrVect;
			
			rbminmax = (refrVect>0.0f)?rbmax:rbmin;
			
			fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			
			posonbox = In.PosW + refrVect*fa;
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
			    	refrVect = refract(-Vn, Nn , refractionIndex[r]);
			    	refrColor += cubeTexRefl.Sample(g_samLinear,refrVect)* colors[r];
				}
		}
		reflColor = lerp(refrColor,reflColor,fresRefl);
		
	
		float inverseDotView = 1.0 - max(dot(Nn,Vn),0.0);
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
	
	for(int i = 0; i< numLights; i++){
		
		
		float4 lightToObject = float4(lPos[i],1) - float4(In.PosW,1);
		float lightDist = length(lightToObject);
		float falloff = lightRange[i%numLighRange]-length(lightToObject);
		
		
		switch (lightType[i]){
			case 0:
			
				lightCounter ++;

				if(useShadow[i] == 1){
					
					shadowCounter++;
				
					viewPosition = mul(float4(In.PosW,1), LightVP[i]);
					
					projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
		
					if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)){							
			
					//		float shadowMapDepth = shadowMap.Sample(shadowSampler, float3(projectTexCoord, shadowCounter-1), 0 ).x;
//			
//						shadowMapDepth += shadowMapBias + 0.001;
//					
//						if ( (shadowMapDepth) < viewPosition.z){
//							ambient += lAmbient[i%numlAmb];
//							break;
//						} 

					
							
							LightDirV = mul(normalize(lightToObject), tV);
							newCol += PhongDirectional(In.PosO, In.ViewDirV, LightDirV, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity).rgb;
//							newCol = min(shadowed,newCol);
//							newCol *= saturate(shadowed+.5);
//							newCol *= calcShadow(NormV,viewPosition,projectTexCoord,shadowCounter);
							newCol *= calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1);					
					} 
					else {
						LightDirV = mul(normalize(lightToObject), tV);
						newCol += PhongDirectional(NormV, In.ViewDirV, LightDirV, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity).rgb;
					}
				} else {
					LightDirV = mul(normalize(lightToObject), tV);
					newCol += PhongDirectional(NormV, In.ViewDirV, LightDirV, lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity).rgb;
				}
				ambient += saturate(lAmbient[i%numlAmb]);
			
				break;
	
			case 1:
				
				lightCounter ++;
				
				if(useShadow[i]  == 1){
					shadowCounter++;
				} 

				viewPosition = mul(float4(In.PosW,1), LightVP[i]);
					
				projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
		   		projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;			
			
				if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)){
					
					
					projectionColor = lightMap.Sample(g_samLinear, float3(projectTexCoord, i), 0 );
					projectionColor *= saturate(1/(viewPosition.z*spotFade));
					

					LightDirW = normalize(lightToObject);
					LightDirV = mul(LightDirW, tV);
					if(useShadow[i]){

						projectionColor *= calcShadowVSM(lightDist,projectTexCoord,shadowCounter-1);
			
					}
			  		newCol += PhongPointSpot(lightDist, NormV, In.ViewDirV, LightDirV, lPos[i],
							  lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2], lDiff[i%numlDiff],
							  lSpec[i%numlSpec],specIntensity, projectTexCoord,projectionColor,lightRange[i%numLighRange]).rgb;
						
				}
			
				ambient += saturate(lAmbient[i%numlAmb]*falloff);
				
				break;
	
			case 2:
				
				
				bool shadowed = false;
			
				lightCounter+=6;
				float4 shadow = 1;
				
				if(useShadow[i]){
						
					shadowCounter+=6;
					for(int p = 0; p < 6; p++){

						viewPosition = mul(float4(In.PosW,1), LightVP[p + lightCounter-6]);
				
						projectTexCoord.x =  viewPosition.x / viewPosition.w / 2.0f + 0.5f;
			   			projectTexCoord.y = -viewPosition.y / viewPosition.w / 2.0f + 0.5f;
					
						if((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y)){

//							float shadowMapDepth = shadowMap.Sample(shadowSampler, float3(projectTexCoord,p+shadowCounter-6),0 ).x;
						
//							shadowMapDepth = LightP[i]._43/(shadowMapDepth-LightP[i]._33);
//							shadowMapDepth += shadowMapBias;
//
//							if ( (shadowMapDepth) < viewPosition.z ) shadowed = true;
		
							shadow -= min(shadow, calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));
//							shadow = calcShadowVSM(lightDist,projectTexCoord,8);
							
//							shadow = calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6);
//							saturate(shadow);
//							newCol *= (calcShadowVSM(lightDist,projectTexCoord,p+shadowCounter-6));
						//	newCol = 1;
						} 
						
					}
				
//					if ( shadowed){
//						ambient += saturate(lAmbient[i%numlAmb]*falloff);
//						break;
//					} 
					
				} 
			
				LightDirW = normalize(lightToObject);
				LightDirV = mul(LightDirW, tV);
				ambient += saturate(lAmbient[i%numlAmb]*falloff);
		  		newCol += PhongPoint(lightDist, NormV, In.ViewDirV, LightDirV, lPos[i],
						  			 lAtt0[i%numlAtt0],lAtt1[i%numlAtt1],lAtt2[i%numlAtt2],
									 lDiff[i%numlDiff], lSpec[i%numlSpec],specIntensity,
									 lightRange[i%numLighRange]).rgb * saturate(shadow);
				//newCol = shadow;
				
			break;
			
		}
		
		
		
	}
	
	
	float3 newRefl = (reflColor+iridescenceColor)*reflective.x*saturate(specIntensity) + RimColor * fresRim;
//	float3 finalDiffuse = (saturate(newCol) + lAmb.rgb + reflColorNorm * reflective.y) * texCol;
	float3 finalDiffuse = saturate(saturate(newCol) + saturate(ambient) + reflColorNorm * reflective.y);

	newCol += newRefl;
	if(refraction) fresRefl = 1;
	newCol += finalDiffuse*(1 - reflective.x * fresRefl );	

	
    return (float4(newCol, Alpha)) * Color.rgba * texCol;

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
