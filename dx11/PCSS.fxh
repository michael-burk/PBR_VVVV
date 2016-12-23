
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
               		//	shadowMap.Sample(shadowSampler, float3(projectTexCoord, shadowCounter-1), 0 ).x;
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

float PCF_Filter(float3 uv, float4 LP, SamplerState ShadowMapSampler, 
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
                       
                       shadMapDepth = shadowMap.Sample(ShadowMapSampler, float3( (uv.xy + float2(i*stepSize,j*stepSize)), uv.z), 0 ).x;
               	
               			//= tex2D(ShadowMap, uv + float2(i*stepSize,j*stepSize)).x;			
               	
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