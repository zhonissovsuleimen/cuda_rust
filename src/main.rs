use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use minifb::{Key, Window, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

#[cuda_module]
mod kernels {
  use super::*;

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
      *c_elem = a[idx_raw] + b[idx_raw];
    }
  }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let ctx = CudaContext::new(0)?;
  let stream = ctx.default_stream();

  const N: usize = 1024;
  let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
  let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

  let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
  let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
  let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

  // SAFETY: this package owns the embedded device bundle produced for the
  // kernels module above.
  let module = unsafe { kernels::load(&ctx)? };
  let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
  module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

  let c_host = c_dev.to_host_vec(&stream)?;
  let errors = (0..N)
    .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
    .count();

  if errors == 0 {
    println!("PASSED: all {} elements correct", N);
  } else {
    eprintln!("FAILED: {} errors", errors);
    std::process::exit(1);
  }

  let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

  let mut window = Window::new(
    "Test - ESC to exit",
    WIDTH,
    HEIGHT,
    WindowOptions::default(),
  )
  .unwrap_or_else(|e| {
    panic!("{}", e);
  });

  // Limit to max ~60 fps update rate
  window.set_target_fps(60);

  while window.is_open() && !window.is_key_down(Key::Escape) {
    for i in buffer.iter_mut() {
      *i = 0; // write something more funny here!
    }

    // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
    window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
  }
  Ok(())
}
