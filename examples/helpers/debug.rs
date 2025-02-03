use std::thread::{self, JoinHandle};

use glam::{vec4, Vec3, Vec4, Vec4Swizzles};
use minifb::{Key, Window, WindowOptions};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

/// Spawn a debug window in a separate thread.
#[allow(dead_code)]
pub fn debug_window<F>(
    width: usize,
    height: usize,
    options: WindowOptions,
    draw: F,
) -> JoinHandle<()>
where
    F: Fn(&mut Window, &mut [u32]) + Send + 'static,
{
    thread::spawn(move || {
        let mut window = Window::new("", width, height, options).unwrap();
        window.set_target_fps(30);
        let mut buffer = vec![0u32; width * height];
        while window.is_open() && !window.is_key_down(Key::Escape) {
            draw(&mut window, &mut buffer);
            window.update_with_buffer(&buffer, width, height).unwrap();
        }
    })
}

/// Spawn a simple debug window in a separate thread. Shared buffer is drawn directly to window.
#[allow(dead_code)]
pub fn simple_debug_window(
    width: usize,
    height: usize,
) -> (AtomicColorBuffer, std::thread::JoinHandle<()>) {
    let shared_buffer = AtomicColorBuffer::new(width, height);
    let window_buffer = shared_buffer.clone();
    let window_thread = debug_window(width, height, Default::default(), move |_window, buffer| {
        for (i, pixel) in buffer.iter_mut().enumerate() {
            *pixel = color_to_minifb_pixel(shared_buffer.get(i));
        }
    });
    (window_buffer, window_thread)
}

/// A very basic buffer for async debug rendering
#[derive(Clone)]
#[allow(dead_code)]
pub struct AtomicColorBuffer {
    pub data: Arc<Vec<[AtomicU32; 4]>>,
    pub width: usize,
    pub height: usize,
}
impl AtomicColorBuffer {
    #[allow(dead_code)]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: Arc::new(
                (0..width * height)
                    .map(|_| [const { AtomicU32::new(0) }; 4])
                    .collect::<Vec<_>>(),
            ),
            width,
            height,
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, i: usize) -> Vec4 {
        vec4(
            f32::from_bits(self.data[i][0].load(Ordering::Relaxed)),
            f32::from_bits(self.data[i][1].load(Ordering::Relaxed)),
            f32::from_bits(self.data[i][2].load(Ordering::Relaxed)),
            f32::from_bits(self.data[i][3].load(Ordering::Relaxed)),
        )
    }

    #[allow(dead_code)]
    pub fn set(&self, i: usize, color: Vec4) {
        self.data[i][0].store(color.x.to_bits(), Ordering::Relaxed);
        self.data[i][1].store(color.y.to_bits(), Ordering::Relaxed);
        self.data[i][2].store(color.z.to_bits(), Ordering::Relaxed);
        self.data[i][3].store(color.w.to_bits(), Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn get_px(&self, x: usize, y: usize) -> Vec4 {
        self.get(y * self.width + x)
    }

    #[allow(dead_code)]
    pub fn set_px(&self, x: usize, y: usize, color: Vec4) {
        self.set(y * self.width + x, color)
    }
}

#[allow(dead_code)]
pub fn color_to_minifb_pixel(color: Vec4) -> u32 {
    let c = (color.xyz().clamp(Vec3::ZERO, Vec3::ONE) * 255.0).as_uvec3();
    (c.x << 16) | (c.y << 8) | c.z
}
