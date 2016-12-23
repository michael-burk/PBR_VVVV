float noise = .001;
	static float2 poissonDisk[16] =
	{
		float2(0.2770745f, 0.6951455f),
		float2(0.1874257f, -0.02561589f),
		float2(-0.3381929f, 0.8713168f),
		float2(0.5867746f, 0.1087471f),
		float2(-0.3078699f, 0.188545f),
		float2(0.7993396f, 0.4595091f),
		float2(-0.09242552f, 0.5260149f),
		float2(0.3657553f, -0.5329605f),
		float2(-0.3829718f, -0.2476171f),
		float2(-0.01085108f, -0.6966301f),
		float2(0.8404155f, -0.3543923f),
		float2(-0.5186161f, -0.7624033f),
		float2(-0.8135794f, 0.2328489f),
		float2(-0.784665f, -0.2434929f),
		float2(0.9920505f, 0.0855163f),
		float2(-0.687256f, 0.6711345f)
	};

float random(float3 seed, int i){
	float4 seed4 = float4(seed,i);
	float dot_product = dot(seed4, float4(12.9898,78.233,45.164,94.673));
	return frac(sin(dot_product) * 43758.5453);
}
float _dnoise1(float3 u){
	u=dot(u+.2,float3(1,57,21));
	return (u.x*(.1+sin(u.x)));
}
float4 _dnoise4(float2 x,float RandomSeed){
	RandomSeed+=.00001;
	float4 c={
	_dnoise1(float3((x+RandomSeed*13+41)+11,length(sin((x-59)/151+RandomSeed*float2(11,7))))+.5),
	_dnoise1(float3((x+RandomSeed*7+293)+5,length(sin((x+127)/163+RandomSeed*float2(13,5))))+.5),
	_dnoise1(float3((x+RandomSeed*5+113)+7,length(sin((x+191)/173+RandomSeed*float2(7,17))))+.5),
	_dnoise1(float3((x+RandomSeed*11+97)+13,length(sin((x-37)/181+RandomSeed*float2(5,23))))+.5)
	};
	return frac(c+x.x*2+RandomSeed+dot(c,1));
}

float2 rand(float2 coord) //generating noise/pattern texture for dithering
{
//	float noiseX = ((frac(1.0-coord.x*(R.x/2.0))*0.25)+(frac(coord.y*(R.y/2.0))*0.75))*2.0-1.0;
//	float noiseY = ((frac(1.0-coord.x*(R.x/2.0))*0.75)+(frac(coord.y*(R.y/2.0))*0.25))*2.0-1.0;
	

	float	noiseX = clamp(frac(sin(dot(coord ,float2 (12.9898,78.233))) * 43758.5453),0.0,1.0)*2.0-1.0;
	float	noiseY = clamp(frac(sin(dot(coord ,float2 (12.9898,78.233)*2.0)) * 43758.5453),0.0,1.0)*2.0-1.0;

	return float2(noiseX,noiseY);
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

// ----------------------------------------------------
// Step 3: Percentage-closer filter implementation with
// variable filter width and number of samples.
// This assumes a square filter with the same number of
// horizontal and vertical samples.
// ----------------------------------------------------

float PCF_Filter(float3 seed, float3 uv, float4 LP, SamplerState ShadowMapSampler, 
                uniform float bias, float filterWidth, float numSamples)
{
       // compute step size for iterating through the kernel
       float stepSize = 2 * filterWidth / numSamples;

       // compute uv coordinates for upper-left corner of the kernel
      // uv = uv - float2(filterWidth,filterWidth);
	 
		uv = float3( (uv.xy - float2(filterWidth, filterWidth)), uv.z );
     
	float sum = 0;  // sum of successful depth tests

       // now iterate through the kernel and filter
       for (int i=0; i<numSamples; i++) {
               for (int j=0; j<numSamples; j++) {
                       // get depth at current texel of the shadow map
                       float shadMapDepth = 0;
               	
//               		 int index = (_dnoise4(seed,seed.z).x*16)%16;
               			 int index = (rand(seed.xy))*16%16;
//               		 int index = seed.xy;
//                       shadMapDepth = shadowMap.Sample(ShadowMapSampler, float3( (uv.xy + float2(i*stepSize,j*stepSize) ), uv.z )).x;
               			 shadMapDepth = shadowMap.Sample(ShadowMapSampler, float3( (uv.xy + poissonDisk[index]*noise+float2(i*stepSize,j*stepSize) ), uv.z )).x;
               			
                       // test if the depth in the shadow map is closer than
                       // the eye-view point
                       float shad = LP.z < shadMapDepth;

                       // accumulate result
                       sum += shad;
               		//	sum = 1;
               }
       }
       
       // return average of the samples
       return sum / (numSamples*numSamples);
		//return 1;
}