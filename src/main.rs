use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use minifb::{Key, Window, WindowOptions};

use crate::{
  consts::{HEIGHT, SIZE, WIDTH},
  cuda::kernels,
};

mod consts;
mod cuda;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let timer = Instant::now();
  let mut buffer_host: Vec<u32> = vec![0; WIDTH * HEIGHT];

  let mut window = Window::new(
    "Test - ESC to exit",
    WIDTH,
    HEIGHT,
    WindowOptions {
      resize: true,
      scale_mode: minifb::ScaleMode::Stretch,
      ..Default::default()
    },
  )?;

  window.set_target_fps(60);

  let ctx = CudaContext::new(0)?;
  let stream = ctx.default_stream();
  let launch_config = LaunchConfig1D::new((SIZE as u32).div_ceil(256), 256, 0);

  let mut buffer_device = DeviceBuffer::from_host(&stream, &buffer_host)?;

  let module = unsafe { kernels::load(&ctx)? };
  while window.is_open() && !window.is_key_down(Key::Escape) {
    let prepared = module.prepare_redraw(launch_config)?;

    module.redraw(
      &stream,
      &prepared,
      &mut buffer_device,
      timer.elapsed().as_secs_f32(),
    )?;

    buffer_host = buffer_device.to_host_vec(&stream)?;

    window
      .update_with_buffer(&buffer_host, WIDTH, HEIGHT)
      .unwrap();
  }
  Ok(())
}
