use argh::FromArgs;
use glam::*;
use image::{ImageBuffer, Rgba};
use obvhs::{
    BvhBuildParams,
    cwbvh::builder::build_cwbvh_from_tris,
    ray::{Ray, RayHit},
    triangle::Triangle,
};
use std::{path::PathBuf, thread, time::Duration};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[path = "./helpers/debug.rs"]
mod debug;
use debug::simple_debug_window;
#[path = "./helpers/load_obj.rs"]
mod load_obj;
use load_obj::load_obj_mesh_data;

use crate::debug::AtomicColorBuffer;

#[derive(FromArgs)]
/// `demoscene` example
struct Args {
    /// if set, no window is created to show render progress.
    #[argh(switch)]
    no_window: bool,
    /// image resolution width (image height is also derived from this).
    #[argh(option, default = "1280")]
    width: usize,
    /// obj file to load (Optionally compressed with zstd. Ex: "file.obj.zst")
    #[argh(
        option,
        short = 'i',
        default = "std::path::PathBuf::from(\"assets/kitchen.obj.zst\")"
    )]
    file: PathBuf,
    /// save output render to file as png
    #[argh(option, short = 'o')]
    output: Option<String>,
}

fn main() {
    let args: Args = argh::from_env();
    render(args, BvhBuildParams::fast_build());
}

fn render(args: Args, bvh_build_params: BvhBuildParams) -> Vec<Vec3A> {
    let tris = load_obj_mesh_data(&args.file)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // Build cwbvh (Change this to build_bvh2_from_tris to try with Bvh2)
    let bvh = build_cwbvh_from_tris(&tris, bvh_build_params, &mut Duration::default());

    // The reason for this mapping below is that if multiple primitives are contained in a cwbvh node, they need to have their indices layed out contiguously.
    // If we want to avoid this indirection during traversal there are two options:
    // 1. Layout the primitives in the order of the cwbvh's indices mapping so that this can index directly into the primitive list.
    // 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    let bvh_tris = bvh
        .primitive_indices
        .iter()
        .map(|i| tris[*i as usize])
        .collect::<Vec<Triangle>>();

    // Setup render target and camera
    let width = args.width;
    let height = ((width as f32) * 0.5625) as usize;
    let target_size = Vec2::new(width as f32, height as f32);
    let fov = 90.0f32;
    let eye = vec3a(3.0, 1.5, 1.4);
    let look_at = vec3(-3.9, 1.5, -1.7);

    // Compute camera projection & view matrices
    let aspect_ratio = target_size.x / target_size.y;
    let proj_inv =
        Mat4::perspective_infinite_reverse_rh(fov.to_radians(), aspect_ratio, 0.01).inverse();
    let view_inv = Mat4::look_at_rh(eye.into(), look_at, Vec3::Y).inverse();

    let shared_buffer =
        (!args.no_window).then(|| AtomicColorBuffer::new(width as usize, height as usize));
    let shared_buffer_clone = shared_buffer.clone();

    let render_thread = thread::spawn(move || {
        #[cfg(feature = "parallel")]
        let iter = (0..width * height).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = (0..width * height).into_iter();

        // For each pixel trace ray into scene and write normal as color to image buffer
        iter.map(|i| {
            let frag_coord = uvec2((i % width) as u32, (i / width) as u32);
            let mut screen_uv = frag_coord.as_vec2() / target_size;
            screen_uv.y = 1.0 - screen_uv.y;
            let ndc = screen_uv * 2.0 - Vec2::ONE;
            let clip_pos = vec4(ndc.x, ndc.y, 1.0, 1.0);

            let mut vs_pos = proj_inv * clip_pos;
            vs_pos /= vs_pos.w;
            let direction = (Vec3A::from((view_inv * vs_pos).xyz()) - eye).normalize();
            let ray = Ray::new(eye, direction, 0.0, f32::MAX);

            let mut hit = RayHit::none();
            if bvh.ray_traverse(ray, &mut hit, |ray, id| bvh_tris[id].intersect(ray)) {
                let mut normal = bvh_tris[hit.primitive_id as usize].compute_normal();
                normal *= normal.dot(-ray.direction).signum(); // Double sided
                if let Some(shared_buffer_clone) = &shared_buffer_clone {
                    shared_buffer_clone.set(i, normal.extend(0.0));
                }
                normal
            } else {
                Vec3A::ZERO
            }
        })
        .collect::<Vec<_>>()
    });

    let fragments = render_thread.join().unwrap();

    if let Some(shared_buffer) = shared_buffer {
        simple_debug_window(width, height, shared_buffer); // Wait for window to close.
    }

    if let Some(output) = args.output {
        // Init image buffer
        let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);
        let pixels = img.as_mut();
        // Convert normals to rgb8 color
        pixels.chunks_mut(4).enumerate().for_each(|(i, chunk)| {
            let c = (fragments[i].clamp(Vec3A::ZERO, Vec3A::ONE) * 255.0).as_uvec3();
            chunk.copy_from_slice(&[c.x as u8, c.y as u8, c.z as u8, 255]);
        });
        img.save(output).expect("Failed to save image");
    }

    fragments
}

#[cfg(test)]
mod tests {
    use super::*;
    use obvhs::test_util::sampling::hash_vec3a_vec;

    fn test_with_build_params(bvh_build_params: BvhBuildParams) {
        assert_eq!(
            hash_vec3a_vec(&render(
                Args {
                    no_window: true,
                    width: 32,
                    file: PathBuf::from("assets/kitchen.obj.zst"),
                    output: None,
                },
                bvh_build_params
            ),),
            1343358762
        );
    }

    #[test]
    fn test_fastest() {
        test_with_build_params(BvhBuildParams::fastest_build())
    }
    #[test]
    fn test_fast() {
        test_with_build_params(BvhBuildParams::fast_build())
    }
    #[test]
    fn test_medium() {
        test_with_build_params(BvhBuildParams::medium_build())
    }
    #[test]
    fn test_slow() {
        test_with_build_params(BvhBuildParams::slow_build())
    }
    #[test]
    fn test_very_slow() {
        test_with_build_params(BvhBuildParams::very_slow_build())
    }
}
