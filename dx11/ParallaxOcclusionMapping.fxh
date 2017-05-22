float fHeightMapScale = -.1;
int nMaxSamples = 10;
int nMinSamples = 1;

float3 parallaxOcclusionMapping(float2 texcoord, float3 V, float3 N){
    
    float fParallaxLimit = -length( V.xy ) / V.z;
    fParallaxLimit *= fHeightMapScale;  
    
    float2 vOffsetDir = normalize( V.xy );
    float2 vMaxOffset = vOffsetDir * fParallaxLimit;
    
    int nNumSamples = (int)lerp( nMaxSamples, nMinSamples, saturate(-dot( V, N )) );
//  nNumSamples = 20;
    float fStepSize = 1.0 / (float)nNumSamples;
    
    float2 dx = ddx( texcoord );
    float2 dy = ddy( texcoord );
    
    float fCurrRayHeight = 1.0;
    float2 vCurrOffset = float2( 0, 0 );
    float2 vLastOffset = float2( 0, 0 );
    
    float fLastSampledHeight = 1;
    float fCurrSampledHeight = 1;

    int nCurrSample = 0;
    
    
    while ( nCurrSample < nNumSamples ){    
                
      fCurrSampledHeight = heightMap.SampleGrad( g_samLinear, texcoord + vCurrOffset, dx, dy ).r;
      if ( fCurrSampledHeight > fCurrRayHeight ){
        float delta1 = fCurrSampledHeight - fCurrRayHeight;
        float delta2 = ( fCurrRayHeight + fStepSize ) - fLastSampledHeight;
    
        float ratio = delta1/(delta1+delta2);
    
        vCurrOffset = (ratio) * vLastOffset + (1.0-ratio) * vCurrOffset;
    
        nCurrSample = nNumSamples + 1;
      } else {
        nCurrSample++;
    
        fCurrRayHeight -= fStepSize;
    
        vLastOffset = vCurrOffset;
        vCurrOffset += fStepSize * vMaxOffset;
    
        fLastSampledHeight = fCurrSampledHeight;
      }
    
    }
//    return texcoord + vCurrOffset;  
    return float3(vCurrOffset,fLastSampledHeight*fHeightMapScale);  
}