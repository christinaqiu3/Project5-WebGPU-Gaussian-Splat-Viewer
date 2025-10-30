import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  render_settings_buffer: GPUBuffer,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data as BufferSource);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const zero_buffer = createBuffer(
    device,
    'zero buffer',
    4,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    nulling_data
  );

  const render_settings_buffer = createBuffer(
    device,
    'render settings buffer',
    8,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg]) 
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const camera_bind_group = device.createBindGroup({
    label: 'camera bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
    ],
  });

  const gaussian_preprocess_bind_group = device.createBindGroup({
    label: 'gaussian preprocess bind group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
    ],
  });

  var splat_size = 0;
  splat_size += 16; // for position, size, color
  splat_size *= pc.num_points; 

  const splat_buffer = createBuffer(
    device,
    'splat buffer',
    splat_size,
    GPUBufferUsage.STORAGE,
    null
  );

  const compute_bind_group = device.createBindGroup({
    label: 'compute bind group',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },//buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer}},
    ],
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  // instance quads for each point in cloud, compute pipeline and then rendering pipeline
  // use create buffer function tomake vertex and index buffer (using gpu buffer usage, buffer usage.indirect)
  // pc.num_points number of quads

  const indirect_buffer = createBuffer(
    device,
    'indirect draw buffer',
    4 * 4, // 4 uint32 values
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    new Uint32Array([6, 0, 0, 0]) // 6 vertices per quad, num_points instances
  );

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render pipeline',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'vs_main',
      buffers: [],
    },
    fragment: {
      module: device.createShaderModule({ code: renderWGSL }),
      entryPoint: 'fs_main',
      targets: [
        { format: presentation_format, 
          blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        } 
      },
      ],
    },
  });
  
  const render_bind_group = device.createBindGroup({
    label: 'render bind group',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: camera_buffer}},
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const compute = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_preprocess_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    pass.setBindGroup(3, compute_bind_group);
    pass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));
    pass.end();
  }

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      
      // clear sorter buffers
      encoder.copyBufferToBuffer(
        zero_buffer,
        0,
        sorter.sort_info_buffer,
        0,
        4
      );
      encoder.copyBufferToBuffer(
        zero_buffer,
        0,
        sorter.sort_dispatch_indirect_buffer,
        0,
        4
      );
      
      // Preprocess compute pass
      compute(encoder);
      
      sorter.sort(encoder);

      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer,
        0,
        indirect_buffer,
        4,
        4
      );

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: texture_view,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      pass.setPipeline(render_pipeline);
      pass.setBindGroup(0, render_bind_group);
      pass.drawIndirect(indirect_buffer, 0);
      pass.end();

    },
    camera_buffer,
    render_settings_buffer,
  };
}
