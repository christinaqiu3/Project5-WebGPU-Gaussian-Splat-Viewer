const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos: vec4<f32>,
    scale: vec2<f32>,
    color: array<u32, 2>,
    conic_opacity: array<u32, 2>,
};

//TODO: bind your data here
// camera uniforms
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
// render settings
// @group(0) @binding(1)
// var<uniform> render_settings: RenderSettings;
// gaussians from load.ts
@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
// splats to be filled in preprocess compute shader
// @group(1) @binding(1)
// var<storage, read_write> splats : array<Splat>;
// sorting related buffers
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
// storage for splats
@group(3) @binding(0)
var<storage, read_write> splats: array<Splat>;
// render settings
@group(3) @binding(1)
var<uniform> render_settings: RenderSettings;
@group(3) @binding(2) // 2
var<storage, read> colors: array<u32>;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored

    let index = splat_idx * 24 + (c_idx / 2) * 3 + c_idx % 2;

    if (c_idx%2 == 0) {
        let unpacked_a = unpack2x16float(colors[index]);
        let unpacked_b = unpack2x16float(colors[index + 1]);
        return vec3f(unpacked_a.x, unpacked_a.y, unpacked_b.x);
    } else {
        let unpacked_a = unpack2x16float(colors[index]);
        let unpacked_b = unpack2x16float(colors[index + 1]);
        return vec3f(unpacked_a.y, unpacked_b.x, unpacked_b.y);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    // TODO: set up pipeline as described in instruction

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    let opacity = a.y;

    // clip position
    let clip_pos = camera.proj * camera.view * pos;
    // 
    let ndc_pos = clip_pos.xyz / clip_pos.w;    
    
    let view_depth = (camera.view * pos).z;
    // frustum culling
    if (ndc_pos.x < -1.2 || ndc_pos.x > 1.2 || ndc_pos.y < -1.2 || ndc_pos.y > 1.2 || view_depth < 0.0) {
        return;
    }

    let rot_a = unpack2x16float(gaussian.rot[0]);
    let rot_b = unpack2x16float(gaussian.rot[1]);
    let rot = vec4<f32>(rot_a.x, rot_a.y, rot_b.x, rot_b.y);
    let rotMat = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.x * rot.y - rot.w * rot.z), 2.0 * (rot.x * rot.z + rot.y * rot.w)),
        vec3<f32>(2.0 * (rot.x * rot.y + rot.w * rot.z), 1.0 - 2.0 * (rot.x * rot.x + rot.w * rot.w), 2.0 * (rot.y * rot.z - rot.x * rot.w)),
        vec3<f32>(2.0 * (rot.x * rot.z - rot.y * rot.w), 2.0 * (rot.y * rot.z + rot.x * rot.w), 1.0 - 2.0 * (rot.x * rot.x + rot.y * rot.y))
    );

    let scale_a = unpack2x16float(gaussian.scale[0]);
    let scale_b = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(scale_a.x, scale_a.y, scale_b.x));

    let scaleMat = mat3x3<f32>(
        vec3<f32>(scale.x * render_settings.gaussian_scaling, 0.0, 0.0),
        vec3<f32>(0.0, scale.y * render_settings.gaussian_scaling, 0.0),
        vec3<f32>(0.0, 0.0, scale.z * render_settings.gaussian_scaling)
    );

// covariance matrix
    let covMat = transpose(scaleMat * rotMat) * (scaleMat * rotMat);

    let covariance = array<f32,6>(
        covMat[0][0],
        covMat[0][1],
        covMat[0][2],
        covMat[1][1],
        covMat[1][2],
        covMat[2][2]
    );

// need to get t vector, j mat, w mat, t mat, v mat, to get 2d cov mat and cov
    let temp = (camera.view * pos).xyz; 
    var t = temp;

    let maxTx = max(.65f * camera.viewport.x / camera.focal.x, temp.x/temp.z);
    let maxTy = max(.65f * camera.viewport.y / camera.focal.y, temp.y/temp.z);
    t.x = min(.65f * camera.viewport.x / camera.focal.x, maxTx) * temp.z;
    t.y = min(.65f * camera.viewport.y / camera.focal.y, maxTy) * temp.z;

    let jMat = mat3x3f(
        camera.focal.x/t.z, 0.f, -(camera.focal.x * t.x) / (t.z * t.z),
        0.f, camera.focal.y/t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.f, 0.f, 0.f
    );

    let wMat = transpose(mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));

    let tMat = wMat * jMat;

    let vMat = mat3x3f(
        covariance[0],
        covariance[1],
        covariance[2],
        covariance[1],
        covariance[3],
        covariance[4],
        covariance[2],
        covariance[4],
        covariance[5],
        
    );

    var covMat2D = transpose(tMat) * transpose(vMat) * tMat;
    covMat2D[0][0] += .3f;
    covMat2D[1][1] += .3f;
    // why add?

    let covariance2D = vec3f(covMat2D[0][0], covMat2D[0][1], covMat2D[1][1]);


    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    // TODO: compute splat size based on depth
    var determinant = covariance2D.x * covariance2D.z - covariance2D.y * covariance2D.y;
    if (determinant == 0.f) { return; }

    let middle = (covariance2D.x + covariance2D.z) * .5f;
    let lambda1 = middle + sqrt(max(.1f, middle * middle - determinant));
    let lambda2 = middle - sqrt(max(.1f, middle * middle - determinant));
    let rad = ceil(3.f * sqrt(max(lambda1, lambda2)));

    let size = vec2f(rad, rad) / camera.viewport;
    
    let index = atomicAdd(&sort_infos.keys_size, 1u);
    splats[index].pos = vec4<f32>(ndc_pos.x, ndc_pos.y, ndc_pos.z, 1.0);
    splats[index].scale = size;


    let direction = normalize(pos.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(direction, idx, u32(render_settings.sh_deg));

    splats[index].color[0] = pack2x16float(color.rg);
    splats[index].color[1] = pack2x16float(vec2f(color.b, 1.f));
    
    // conic
    let conic = vec3f(covariance2D.z, -covariance2D.y, covariance2D.x) / determinant;

    splats[index].conic_opacity[0] = pack2x16float(conic.xy);
    splats[index].conic_opacity[1] = pack2x16float(vec2f(conic.z, 1.f / (1.f + exp(-opacity))));
    


    sort_depths[index] = bitcast<u32>(100.0f - view_depth);
    sort_indices[index] = index;

    if (index % keys_per_dispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}