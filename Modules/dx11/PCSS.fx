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


// -------------------------------------
// STEP 1: Search for potential blockers
// -------------------------------------
float findBlocker(
		float3 uv,
		float4 LP,
		SamplerState ShadowMapSampler,
		float bias,
		float searchWidth,
		float numSamples)
{
        // divide filter width by number of samples to use
        float stepSize = 2 * searchWidth / numSamples;

        // compute starting point uv coordinates for search
        uv = float3( (uv.xy - float2(searchWidth, searchWidth)), uv.z );

        // reset sum to zero
        float blockerSum = 0;
        float receiver = LP.z;
        float blockerCount = 0;
        float foundBlocker = 0;

        // iterate through search region and add up depth values
        for (int i=0; i<numSamples; i++) {
               for (int j=0; j<numSamples; j++) {

                       float shadMapDepth = shadowMap.Sample(ShadowMapSampler, float3( (uv.xy + float2(i*stepSize,j*stepSize)), uv.z), 0 ).x;
                       // found a blocker
                       if (shadMapDepth < receiver) {
                               blockerSum += shadMapDepth;
                               blockerCount++;
                               foundBlocker = 1;
                       }
               }
        }

		float result;
		
		if (foundBlocker == 0) {
			// set it to a unique number so we can check
			// later to see if there was no blocker
			result = 999;
		}
		else {
		    // return average depth of the blockers
			result = blockerSum / blockerCount;
		}
		
		return result;
}

// ------------------------------------------------
// STEP 2: Estimate penumbra based on
// blocker estimate, receiver depth, and light size
// ------------------------------------------------
float estimatePenumbra(float4 LP,
			float Blocker,
			uniform float LightSize)
{
       // receiver depth
       float receiver = LP.z;
       // estimate penumbra using parallel planes approximation
       float penumbra = (receiver - Blocker) * LightSize / Blocker;
       return penumbra;
}

float4 PS(psInput input): SV_Target
{
	   float zReceiver = viewPosition.z ;
	   float searchWidth = SceneScale * (zReceiver - 1.0) / zReceiver;
	   float blocker = findBlocker(float3(projectTexCoord, shadowCounter-1), viewPosition-float4(0,0,shadowMapBias,0), shadowSampler, shadowMapBias,
	                              SceneScale * LightSize / (viewPosition.z), shadowsearchSamples);
	   
	   //return (blocker*1);  // uncomment to visualize blockers
	   
	   // ---------------------------------------------------------
	   // Step 2: Estimate penumbra using parallel planes approximation
	   float penumbra;  
	   penumbra = estimatePenumbra(viewPosition, blocker, LightSize);
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





