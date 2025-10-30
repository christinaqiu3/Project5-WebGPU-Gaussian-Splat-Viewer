struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
    @location(1) conic_opacity: vec4<f32>,
    @location(2) center: vec2<f32>,
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos: vec4<f32>,
    // opacity: f32,
    // rot: mat2x2<f32>,
    scale: vec2<f32>,
    color: array<u32, 2>,
    conic_opacity: array<u32, 2>,
};

struct Camera {
    viewMat: mat4x4<f32>,
    invViewMat: mat4x4<f32>,
    projMat: mat4x4<f32>,
    invProjMat: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
}

@group(0) @binding(0)
var<storage,read> splats : array<Splat>;
@group(0) @binding(1)
var<storage,read> sort_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: Camera;

@vertex
fn vs_main( 
    @builtin(instance_index) global_instance_index: u32,
    @builtin(vertex_index) local_vertex_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass to fragment shader

    let index = sort_indices[global_instance_index];
    let splat = splats[index];

    let x = splat.pos.x;
    let y = splat.pos.y;
    let w = splat.scale.x * 2.0f; // because scale is half-width
    let h = splat.scale.y * 2.0f;

// array of 6 vec2s using x y w h
// does the order matter?
    let quad = array<vec2<f32>,6>(
        vec2<f32>(x - w, y - h),
        vec2<f32>(x + w, y - h),
        vec2<f32>(x - w, y + h),
        vec2<f32>(x - w, y + h),
        vec2<f32>(x + w, y - h),
        vec2<f32>(x + w, y + h)
    );

// vertex positions using local_vertex_index
    let position = vec4(quad[local_vertex_index], 0.0, 1.0); // splat.pos.z for depth

    let unpacked_a = unpack2x16float(splat.color[0]);
    let unpacked_b = unpack2x16float(splat.color[1]);
    let color = vec4(unpacked_a.x, unpacked_a.y, unpacked_b.x, unpacked_b.y);
    let size = (vec2f(w, h) * .5f + .5f) * camera.viewport.xy;

// conic
    let conic_unpacked_a = unpack2x16float(splat.conic_opacity[0]);
    let conic_unpacked_b = unpack2x16float(splat.conic_opacity[1]);
    let conic = vec3f(conic_unpacked_a.x, conic_unpacked_a.y, conic_unpacked_b.x);
    let opacity = conic_unpacked_b.y;

    var out: VertexOutput;
    out.position = position;
    out.color = color;
    out.conic_opacity = vec4f(conic, opacity);
    out.center = vec2f(x, y);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var pos = (in.position.xy / camera.viewport) * 2.f - 1.f;
    pos.y *= -1.f;

    var offset = (pos.xy - in.center.xy) * camera.viewport * vec2f(-.5f, .5f);

    var power = in.conic_opacity.x * pow(offset.x, 2.f) + in.conic_opacity.z * pow(offset.y, 2.f);
    power = power * -.5f - in.conic_opacity.y * offset.x * offset.y;

    if (power > 0.f) {
        return vec4f(0.f);
    }

    let alpha = min(.99f, in.conic_opacity.w * exp(power));

    return in.color * alpha;
}