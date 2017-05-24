float fHeightMapScale = -.1;
//int nMaxSamples = 50;
//int nMinSamples = 1;
int POM_numSamples <bool visible=false;> = 25;

float3 parallaxOcclusionMapping(float2 texcoord, float3 V, float3 N){
    
    float fParallaxLimit = -length( V.xy ) / V.z;
    fParallaxLimit *= -fHeightMapScale;  
    
    float2 vOffsetDir = normalize( V.xy );
    float2 vMaxOffset = vOffsetDir * fParallaxLimit;
    
//    int POM_numSamples = (int)lerp( nMaxSamples, nMinSamples, saturate(-dot( N, V)) );
    float fStepSize = 1.0 / (float)POM_numSamples;
    
    float2 dx = ddx( texcoord );
    float2 dy = ddy( texcoord );
    
    float fCurrRayHeight = 1.0;
    float2 vCurrOffset = float2( 0, 0 );
    float2 vLastOffset = float2( 0, 0 );
    
    float fLastSampledHeight = 1;
    float fCurrSampledHeight = 1;

    int nCurrSample = 0;
    
    float delta1;
	float delta2;
	float ratio;
    while ( nCurrSample < POM_numSamples ){    
                
      fCurrSampledHeight = heightMap.SampleGrad( g_samLinear, texcoord + vCurrOffset, dx, dy ).r;
      if ( fCurrSampledHeight > fCurrRayHeight ){
        delta1 = fCurrSampledHeight - fCurrRayHeight;
        delta2 = ( fCurrRayHeight + fStepSize ) - fLastSampledHeight;
    
        ratio = delta1/(delta1+delta2);
    
        vCurrOffset = (ratio) * vLastOffset + (1.0-ratio) * vCurrOffset;
    
        nCurrSample = POM_numSamples + 1;
      } else {
        nCurrSample++;
    
        fCurrRayHeight -= fStepSize;
    
        vLastOffset = vCurrOffset;
        vCurrOffset += fStepSize * vMaxOffset;
    
        fLastSampledHeight = fCurrSampledHeight;
      }
    
    }
    return float3(vCurrOffset,delta1*-fHeightMapScale);  
}

//
//float parallaxSoftShadowMultiplier(float3 L, float2 initialTexCoord,
//                                       float initialHeight)
//{
//	float shadowMultiplier	= 0;
//   // calculate lighting only for surface oriented to the light source
//   if(dot(float3(0, 0, 1), L) > 0)
//   {
//   	  float2 dx = ddx( initialTexCoord );
//   	  float2 dy = ddy( initialTexCoord );
//    
//      // calculate initial parameters
//      float numSamplesUnderSurface	= 0;
////      shadowMultiplier	= 0;
////      float numLayers	= mix(maxLayers, minLayers, abs(dot(vec3(0, 0, 1), L)));
//		
//      float layerHeight	= fHeightMapScale / POM_numSamples;
//      float2 texStep	= -fHeightMapScale * L.xy / L.z / POM_numSamples;
//
//      // current parameters
//      float currentLayerHeight	= fHeightMapScale - layerHeight;
//      float2 currentTextureCoords	= initialTexCoord + texStep;
//      float heightFromTexture = heightMap.SampleGrad( g_samLinear, initialTexCoord, dx, dy ).r;
//      int stepIndex	= 0;
//
//      // while point is below depth 0.0 )
//      while(currentLayerHeight > 0)
//      {
//         // if point is under the surface
//         if(heightFromTexture < currentLayerHeight)
//         {
//            // calculate partial shadowing factor
//            numSamplesUnderSurface	+= 1;
//            float newShadowMultiplier	= (currentLayerHeight - heightFromTexture) *
//                                             (1.0 - stepIndex / POM_numSamples);
//            shadowMultiplier	= max(shadowMultiplier, newShadowMultiplier);
//         }
//
//         // offset to the next layer
//         stepIndex	+= 1;
//         currentLayerHeight	-= layerHeight;
//         currentTextureCoords	+= texStep;
//
//         heightFromTexture	= - heightMap.SampleGrad( g_samLinear, currentTextureCoords, dx, dy ).r;
//      }
//
//      // Shadowing factor should be 1 if there were no points under the surface
//      if(numSamplesUnderSurface < 1)
//      {
//         shadowMultiplier = 1;
//      }
//      else
//      {
//         shadowMultiplier = saturate(1 -shadowMultiplier);
//      }
//   }
//   return shadowMultiplier;
//}