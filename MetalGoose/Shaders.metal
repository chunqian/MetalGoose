#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct UpscaleParams {
    float sharpness;
    uint2 inputSize;
    uint2 outputSize;
};

struct FrameBlendParams {
    float t;
    uint2 textureSize;
};

inline half rgb2luma(half3 rgb) {
    return dot(rgb, half3(0.299h, 0.587h, 0.114h));
}

inline half3 clampColor(half3 color) {
    return clamp(color, half3(0.0h), half3(1.0h));
}

vertex VertexOut texture_vertex(uint vertexID [[vertex_id]]) {
    const float4 positions[4] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0)
    };
    const float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    
    VertexOut out;
    out.position = positions[vertexID];
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment half4 texture_fragment(
    VertexOut in [[stage_in]],
    texture2d<half> texture [[texture(0)]]
) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return texture.sample(s, in.texCoord);
}

kernel void mgup1_upscale(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    constant UpscaleParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.outputSize.x || gid.y >= params.outputSize.y) {
        return;
    }
    
    float2 srcPos = float2(gid) * float2(params.inputSize) / float2(params.outputSize);
    uint2 srcCoord = uint2(clamp(srcPos, float2(0.0), float2(params.inputSize) - 1.0));
    
    half4 center = input.read(srcCoord);
    
    if (params.sharpness < 0.01h) {
        output.write(center, gid);
        return;
    }
    
    uint2 inputMax = params.inputSize - 1;
    
    half4 n = input.read(uint2(srcCoord.x, max(0u, srcCoord.y - 1)));
    half4 s = input.read(uint2(srcCoord.x, min(inputMax.y, srcCoord.y + 1)));
    half4 w = input.read(uint2(max(0u, srcCoord.x - 1), srcCoord.y));
    half4 e = input.read(uint2(min(inputMax.x, srcCoord.x + 1), srcCoord.y));
    
    half4 minNeighbor = min(min(n, s), min(w, e));
    half4 maxNeighbor = max(max(n, s), max(w, e));
    
    half4 contrast = maxNeighbor - minNeighbor;
    half4 sharpWeight = half4(params.sharpness) * saturate(1.0h - contrast * 2.0h);
    
    half4 neighbors = (n + s + w + e) * 0.25h;
    half4 sharpened = center + (center - neighbors) * sharpWeight;
    
    output.write(half4(clampColor(sharpened.rgb), 1.0h), gid);
}

kernel void mgfg1_blend(
    texture2d<half, access::read> prevFrame [[texture(0)]],
    texture2d<half, access::read> currFrame [[texture(1)]],
    texture2d<half, access::write> output [[texture(2)]],
    constant FrameBlendParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.textureSize.x || gid.y >= params.textureSize.y) {
        return;
    }
    
    half4 prev = prevFrame.read(gid);
    half4 curr = currFrame.read(gid);
    
    half t = half(params.t);
    t = t * t * (3.0h - 2.0h * t);
    
    half4 result = mix(prev, curr, t);
    
    output.write(result, gid);
}

kernel void passthrough_copy(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 size = uint2(output.get_width(), output.get_height());
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }
    
    uint2 inputSize = uint2(input.get_width(), input.get_height());
    uint2 srcCoord = gid;
    if (inputSize.x != size.x || inputSize.y != size.y) {
        float2 scale = float2(inputSize) / float2(size);
        srcCoord = uint2(float2(gid) * scale);
        srcCoord = clamp(srcCoord, uint2(0), inputSize - 1);
    }
    
    output.write(input.read(srcCoord), gid);
}

kernel void fxaa_simple(
    texture2d<half, access::read> input [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = input.get_width();
    uint height = input.get_height();
    if (gid.x >= width || gid.y >= height) return;
    
    const half FXAA_REDUCE_MUL = 1.0h / 8.0h;
    const half FXAA_REDUCE_MIN = 1.0h / 128.0h;
    const half FXAA_SPAN_MAX = 8.0h;
    
    half3 rgbNW = input.read(uint2(max(0u, gid.x - 1), max(0u, gid.y - 1))).rgb;
    half3 rgbNE = input.read(uint2(min(width - 1, gid.x + 1), max(0u, gid.y - 1))).rgb;
    half3 rgbSW = input.read(uint2(max(0u, gid.x - 1), min(height - 1, gid.y + 1))).rgb;
    half3 rgbSE = input.read(uint2(min(width - 1, gid.x + 1), min(height - 1, gid.y + 1))).rgb;
    half3 rgbM = input.read(gid).rgb;
    
    half lumaNW = rgb2luma(rgbNW);
    half lumaNE = rgb2luma(rgbNE);
    half lumaSW = rgb2luma(rgbSW);
    half lumaSE = rgb2luma(rgbSE);
    half lumaM = rgb2luma(rgbM);
    
    half lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    half lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
    half2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    
    half dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25h * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    half rcpDirMin = 1.0h / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    
    dir = min(half2(FXAA_SPAN_MAX), max(half2(-FXAA_SPAN_MAX), dir * rcpDirMin));
    
    half2 texOffset1 = dir * (1.0h / 3.0h - 0.5h);
    half2 texOffset2 = dir * (2.0h / 3.0h - 0.5h);
    
    int2 pos1 = int2(half2(gid) + texOffset1);
    int2 pos2 = int2(half2(gid) + texOffset2);
    
    pos1 = clamp(pos1, int2(0), int2(width - 1, height - 1));
    pos2 = clamp(pos2, int2(0), int2(width - 1, height - 1));
    
    half3 rgbA = (input.read(uint2(pos1)).rgb + input.read(uint2(pos2)).rgb) * 0.5h;
    
    half2 texOffset3 = dir * -0.5h;
    half2 texOffset4 = dir * 0.5h;
    
    int2 pos3 = int2(half2(gid) + texOffset3);
    int2 pos4 = int2(half2(gid) + texOffset4);
    
    pos3 = clamp(pos3, int2(0), int2(width - 1, height - 1));
    pos4 = clamp(pos4, int2(0), int2(width - 1, height - 1));
    
    half3 rgbB = rgbA * 0.5h + (input.read(uint2(pos3)).rgb + input.read(uint2(pos4)).rgb) * 0.25h;
    
    half lumaB = rgb2luma(rgbB);
    
    half3 result = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
    
    output.write(half4(result, 1.0h), gid);
}
