
float lPower <String uiname="Power"; float uimin=0.0;> = 25.0;     //shininess of specular highlight

//phong point function
lightStruct PhongPoint(float lightToObject, float3 NormV, float3 ViewDirV, float3 LightDirV, float3 lightPos, float lAtt0,
				  float lAtt1, float lAtt2, float4 lAmb, float4 lDiff, float4 lSpec, float4 specIntensity, float lRange, float4 sF, lightStruct li)
{

			
		 	//lightStruct result;

		    float atten = 1/(saturate(lAtt0) + saturate(lAtt1) * lightToObject + saturate(lAtt2) * pow(lightToObject, 2));
			
			 float4 amb = lAmb * atten;
   			 amb.a = 1;		
	
		    //halfvector
		    float3 H = normalize(ViewDirV + LightDirV);
		
		    //compute blinn lighting
		    float4 shades = lit(dot(NormV, LightDirV), dot(NormV, H), lPower);
		
		    float4 diff = lDiff * shades.y * atten;
		    diff.a = 1;
		
		    //reflection vector (view space)
		    float3 R = normalize(2 * dot(NormV, LightDirV) * NormV - LightDirV);
		
		    //normalized view direction (view space)
		    float3 V = normalize(ViewDirV);
		
		    //calculate specular light
		    float4 spec = pow(max(dot(R, V),0), lPower) * lSpec;
			
		    spec *= specIntensity;
			
			li.diffuse += saturate(diff * saturate(lRange-lightToObject)) * sF;
			li.reflection += saturate(spec * saturate(lRange-lightToObject)) * sF;
			li.ambient += saturate(amb * saturate(lRange-lightToObject)) * sF;	
	
			return li;
	
}
