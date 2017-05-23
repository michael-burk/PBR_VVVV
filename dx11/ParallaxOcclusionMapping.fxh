float fHeightMapScale = -.1;
//int nMaxSamples = 10;
//int nMinSamples = 1;
int POM_numSamples <bool visible=false;> = 25;

float3 parallaxOcclusionMapping(float2 texcoord, float3 V, float3 N){
    
    float fParallaxLimit = -length( V.xy ) / V.z;
    fParallaxLimit *= -fHeightMapScale;  
    
    float2 vOffsetDir = normalize( V.xy );
    float2 vMaxOffset = vOffsetDir * fParallaxLimit;
    
//    int nNumSamples = (int)lerp( nMaxSamples, nMinSamples, saturate(dot( N, V)) );
//  	nNumSamples = 50;
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
