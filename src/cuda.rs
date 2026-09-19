use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
  use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

  use crate::{
    consts::{HEIGHT, WIDTH},
    particle::Particle,
  };

  use super::*;

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn solve(input: &[Particle], mut output: DisjointSlice<Particle>, dt: f32) {
    let idx = thread::index_1d();
    let i = idx.get();

    if let Some(out) = output.get_mut(idx) {
      let input = &input[i];

      *out = *input;

      out.pos[0] += out.vel[0] * dt;
      out.pos[1] += out.vel[1] * dt;

      if out.pos[0] < 0.0 || out.pos[0] >= WIDTH as f32 {
        out.pos[0] = out.pos[0].clamp(0.0, WIDTH as f32 - 1.0);
        out.vel[0] *= -1.0;
      }

      if out.pos[1] < 0.0 || out.pos[1] >= HEIGHT as f32 {
        out.pos[1] = out.pos[1].clamp(0.0, HEIGHT as f32 - 1.0);
        out.vel[1] *= -1.0;
      }
    }
  }

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn clear(mut screen: DisjointSlice<u32>) {
    let idx = thread::index_1d();

    if let Some(pixel) = screen.get_mut(idx) {
      *pixel = 0;
    }
  }
  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub unsafe fn draw(particles: &[Particle], screen: *mut u32) {
    let i = thread::index_1d().get();

    if let Some(p) = particles.get(i) {
      let pixel_id = p.pos[0] as usize + p.pos[1] as usize * WIDTH;

      unsafe {
        DeviceAtomicU32::from_ptr(screen.add(pixel_id)).store(0x000000FF, AtomicOrdering::Relaxed);
      }
    }
  }
}
