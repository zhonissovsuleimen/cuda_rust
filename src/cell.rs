use cuda_core::DeviceCopy;

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Cell {
  pub count: u32,
  pub vel: [f32; 2],
}

impl Cell {
  pub fn empty() -> Self {
    Self {
      count: 0,
      vel: [0.0, 0.0],
    }
  }

  pub fn clear(&mut self) {
    self.count = 0;
    self.vel = [0.0, 0.0];
  }
}
