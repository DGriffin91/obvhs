use std::{f32::consts::PI, time::Duration};

use std::thread;

use glam::*;
use image::{ImageBuffer, Rgba};
use obvhs::{
    BvhBuildParams, Transformable,
    cwbvh::builder::build_cwbvh_from_tris,
    ray::{Ray, RayHit},
    test_util::geometry::{CUBE, PLANE},
    triangle::Triangle,
};

#[path = "./helpers/debug.rs"]
mod debug;
use debug::simple_debug_window;

use crate::debug::AtomicColorBuffer;

// Generate triangles for cornell box
fn generate_cornell_box() -> Vec<Triangle> {
    let floor = PLANE;
    let mut box1 = CUBE;
    let mut box2 = box1;
    let mut ceiling = floor;
    let mut wall1 = floor;
    let mut wall2 = floor;
    let mut wall3 = floor;
    box1.transform(&Mat4::from_scale_rotation_translation(
        Vec3::splat(0.3),
        Quat::from_rotation_y(-17.5f32.to_radians()),
        vec3(0.33, 0.3, 0.37),
    ));
    box2.transform(&Mat4::from_scale_rotation_translation(
        vec3(0.3, 0.6, 0.3),
        Quat::from_rotation_y(17.5f32.to_radians()),
        vec3(-0.33, 0.6, -0.29),
    ));
    ceiling.transform(&Mat4::from_translation(Vec3::Y * 2.0));
    wall1.transform(&Mat4::from_rotation_translation(
        Quat::from_rotation_x(PI * 0.5),
        vec3(0.0, 1.0, -1.0),
    ));
    wall2.transform(&Mat4::from_rotation_translation(
        Quat::from_rotation_z(-PI * 0.5),
        vec3(-1.0, 1.0, 0.0),
    ));
    wall3.transform(&Mat4::from_rotation_translation(
        Quat::from_rotation_z(-PI * 0.5),
        vec3(1.0, 1.0, 0.0),
    ));
    let mut tris = Vec::new();
    tris.extend(floor);
    tris.extend(box1);
    tris.extend(box2);
    tris.extend(ceiling);
    tris.extend(wall1);
    tris.extend(wall2);
    tris.extend(wall3);
    tris
}

fn main() {
    let tris = generate_cornell_box();
    // Build cwbvh (Change this to build_bvh2_from_tris to try with Bvh2)
    let bvh = build_cwbvh_from_tris(
        &tris,
        BvhBuildParams::medium_build(),
        &mut Duration::default(),
    );

    // The reason for this mapping below is that if multiple primitives are contained in a cwbvh node, they need to have their indices laid out contiguously.
    // If we want to avoid this indirection during traversal there are two options:
    // 1. Layout the primitives in the order of the cwbvh's indices mapping so that this can index directly into the primitive list.
    // 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    let bvh_tris = bvh
        .primitive_indices
        .iter()
        .map(|i| tris[*i as usize])
        .collect::<Vec<Triangle>>();

    // Setup render target and camera
    let width = 1280;
    let height = 720;
    let target_size = Vec2::new(width as f32, height as f32);
    let fov = 90.0f32;
    let eye = vec3a(0.0, 1.0, 2.1);
    let look_at = vec3(0.0, 1.0, 0.0);

    // Compute camera projection & view matrices
    let aspect_ratio = target_size.x / target_size.y;
    let proj_inv =
        Mat4::perspective_infinite_reverse_rh(fov.to_radians(), aspect_ratio, 0.01).inverse();
    let view_inv = Mat4::look_at_rh(eye.into(), look_at, Vec3::Y).inverse();

    let shared_buffer = AtomicColorBuffer::new(width, height);
    let shared_buffer_clone = shared_buffer.clone();
    // Render in separate thread so we can asynchronously update window. (Can't run window in other thread on MacOS)
    let render_thread = thread::spawn(move || {
        // Init image buffer
        let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);
        let pixels = img.as_mut();

        // For each pixel trace ray into scene and write normal as color to image buffer
        pixels.chunks_mut(4).enumerate().for_each(|(i, chunk)| {
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
                let c = (normal * 255.0).as_uvec3();
                chunk.copy_from_slice(&[c.x as u8, c.y as u8, c.z as u8, 255]);
                shared_buffer_clone.set(i, normal.extend(0.0));
            }
        });
        img
    });

    simple_debug_window(width, height, shared_buffer); // Wait for window to close.

    let img = render_thread.join().unwrap();

    img.save("basic_cornell_box_rend.png")
        .expect("Failed to save image");
}
