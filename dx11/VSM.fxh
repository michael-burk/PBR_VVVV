

float4 calcShadowVSM(float worldSpaceDistance, float lightRange, float2 projectTexCoord, int shadowCounter){

	
//	float m22 = LightP[shadowCounter]._m22;
//	float m32 = LightP[shadowCounter]._m32;    
//	
//	float near = (2.0f*m32)/(2.0f*m22-2.0f);
//	float far = ((m22-1.0f)*near)/(m22+1.0);
	
	 float currentDistanceToLight = clamp((worldSpaceDistance - 0 /*nearPlane*/) 
     / (lightRange /*farPlane*/ - 0 /*nearPlane*/), 0, 1);
	
    /////////////////////////////////////////////////////////

    // get blured and blured squared distance to light
	
	float4 shadowCol = shadowMap.SampleLevel(shadowSampler, float3(projectTexCoord, shadowCounter), 0);
	float2 depths = shadowCol.xy;
	
    float M1 = depths.x;
    float M2 = depths.y;
    float M12 = M1 * M1;

    float p = 0.0;
    float lightIntensity = 1;
	float alpha = 0;
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
        lightIntensity = clamp((intensity-lightBleedingLimit[shadowCounter])/ (1.0-lightBleedingLimit[shadowCounter]), 0.0, 1.0);
    	
    	alpha +=  (1 - saturate(shadowCol.a));
    }

    /////////////////////////////////////////////////////////

    float4 resultingColor = float4(float3(lightIntensity,lightIntensity,lightIntensity),1);
	
	return resultingColor+alpha;
	
}