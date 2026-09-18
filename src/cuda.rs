use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
  use crate::consts::WIDTH;

  use super::*;

  #[kernel]
  #[launch_bounds(256)]
  #[launch_contract(domain = 1, block = (256, 1, 1))]
  pub fn redraw(mut buf: DisjointSlice<u32>, time: f32) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(val) = buf.get_mut(idx) {
      let x = idx_raw % WIDTH;
      let y = idx_raw / WIDTH;
      let t = time.fract().round() as usize;

      *val = 0 - ((x + y + t) & 1) as u32;
    }
  }
}
