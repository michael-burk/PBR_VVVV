
float4 calcSSS(float worldSpaceDistance, float2 projectTexCoord, int shadowCounter){
	
	    float currentDistanceToLight = clamp((worldSpaceDistance - nearFarPlane[shadowCounter].x) 
        / (nearFarPlane[shadowCounter].y - nearFarPlane[shadowCounter].x), 0, 1);

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
	float intensity;
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
        intensity = sigma2 / (sigma2 + pow(currentDistanceToLight - M1, 2));

        // reduce light bleeding
        lightIntensity = clamp((intensity-lightBleedingLimit[shadowCounter])/ (1.0-lightBleedingLimit[shadowCounter]), 0.0, 1.0);
    	alpha +=  (1 - saturate(shadowCol.a));
    }

    /////////////////////////////////////////////////////////

    float4 resultingColor = float4(float3(lightIntensity,lightIntensity,lightIntensity),1);
	
//	return resultingColor+alpha;
	return depths.x;
	
}