use core::f32;
use std::time::{Duration, Instant};

use glam::*;
use minifb::{Key, MouseButton, Window, WindowOptions};
use obvhs::{
    aabb::Aabb,
    bvh2::{insertion_removal::SiblingInsertionCandidate, Bvh2},
    cwbvh::bvh2_to_cwbvh::bvh2_to_cwbvh,
    heapstack::HeapStack,
    ploc::{PlocSearchDistance, SortPrecision},
    ray::{Ray, RayHit},
    scope, PrettyDuration,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

#[path = "./helpers/debug.rs"]
mod debug;
use debug::color_to_minifb_pixel;

const NO_BVH: bool = false;

const VERIFY_PAIRS: bool = false;

const AABB_OVERSIZE_FACTOR: f32 = 1.0;

const FPS_LIMIT: usize = 120; // 0 for none

const DT: f32 = 0.015; // Constant so it's deterministic

const COUNT: u32 = 3000;

fn main() {
    let mut physics = PhysicsWorld {
        rebuild_method: BvhRebuid::Full,
        ..Default::default()
    };

    // Setup render target and camera
    let width = 512;
    let height = 512;
    let target_size = Vec2::new(width as f32, height as f32);
    let fov = 55.0f32;
    let eye = vec3a(30.0, 15.0, 30.0);
    let look_at = vec3(0.0, 12.0, 0.0);
    // Compute camera projection & view matrices
    let aspect_ratio = target_size.x / target_size.y;
    let proj_inv =
        Mat4::perspective_infinite_reverse_rh(fov.to_radians(), aspect_ratio, 0.01).inverse();
    let view_inv = Mat4::look_at_rh(eye.into(), look_at, Vec3::Y).inverse();

    for i in 0..COUNT {
        let x = (i as f32 * 0.001).sin() * 1.0;
        let z = (i as f32 * 0.001).cos() * 1.0;
        physics.add_sphere(vec3a(x, i as f32 * 0.25, z), Vec3A::ZERO, 0.5, 0.5);
    }
    physics.bvh_full_rebuild();
    let mut steps = 0;
    let start = std::time::Instant::now();

    let mut spawn = false;

    let mut window = Window::new("", width, height, WindowOptions::default()).unwrap();
    window.set_target_fps(FPS_LIMIT);
    let mut buffer = vec![0u32; width * height];
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.get_mouse_down(MouseButton::Left) {
            if !spawn {
                spawn = true;
                physics.add_sphere_update_bvh(
                    vec3a(0.0, 25.0, 0.0),
                    vec3a(0.0, -100.0, 0.0),
                    3.0,
                    30.0,
                );
            }
        } else {
            spawn = false;
        }

        let physics_start = Instant::now();
        physics_update(&mut physics);
        let physics_end = physics_start.elapsed();
        steps += 1;

        let render_start = Instant::now();
        {
            let bvh = &physics.bvh;
            // Seems to be faster to convert to cwbvh each frame before rendering
            let bvh = bvh2_to_cwbvh(&bvh, 3, true, false);

            scope!("render");

            #[cfg(feature = "parallel")]
            let iter = buffer.par_iter_mut();
            #[cfg(not(feature = "parallel"))]
            let iter = buffer.iter_mut();

            iter.enumerate().for_each(|(i, color)| {
                // TODO there seems to be precision issues with rendering if the sphere are far away from the camera.
                // This issue occurs regardless of using a BVH. Maybe something in the projection or ray calculation?
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
                if bvh.ray_traverse(ray, &mut hit, |ray, id| {
                    let primitive_id = bvh.primitive_indices[id] as usize;
                    let sphere = &physics.items[primitive_id];
                    ray_sphere_intersect(&ray, sphere.position, sphere.radius)
                }) {
                    let primitive_id = bvh.primitive_indices[hit.primitive_id as usize];
                    let sphere = &physics.items[primitive_id as usize];
                    let hit_p = ray.origin + ray.direction * hit.t;
                    let normal = (hit_p - sphere.position).normalize_or_zero();
                    *color = color_to_minifb_pixel(normal.extend(1.0));
                } else {
                    *color = 0; // Clear
                }
            });
            window.update_with_buffer(&buffer, width, height).unwrap();
        }
        let render_end = render_start.elapsed();
        window.set_title(&format!(
            "{}: physics, {}: render",
            PrettyDuration(physics_end),
            PrettyDuration(render_end)
        ));
    }
    let elapsed = start.elapsed();
    println!(
        "{:>8} bench | {} per iter",
        format!("{}", PrettyDuration(elapsed)),
        format!(
            "{}",
            PrettyDuration(Duration::from_secs_f32(
                elapsed.as_secs_f32() / steps as f32
            ))
        ),
    );
    dbg!(physics.bvh.depth(0));
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Pair(pub u64);

impl Pair {
    #[inline]
    pub fn new(a: u32, b: u32) -> Pair {
        // Pair inputs need to be sorted so full pair can be matched with duplicate pairs
        let (a, b) = (a.min(b), a.max(b));
        Pair(((a as u64) << 32u64) | b as u64)
    }
    #[inline]
    pub fn get(&self) -> (u32, u32) {
        (
            ((self.0 >> 32u64) & 0xFFFFFFFF) as u32,
            (self.0 & 0xFFFFFFFF) as u32,
        )
    }
}

#[derive(PartialEq, Eq, Default)]
#[allow(dead_code)]
enum BvhRebuid {
    #[default]
    Full,
    Reinsert,
    RemoveAndInsert,
}

struct PhysicsWorld {
    items: Vec<SphereCollider>,
    bvh: Bvh2,
    bvh_insertion_stack: HeapStack<SiblingInsertionCandidate>,
    temp_aabbs: Vec<Aabb>,
    collision_pairs: Vec<Pair>,
    rebuild_method: BvhRebuid,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self {
            items: Default::default(),
            bvh: Default::default(),
            bvh_insertion_stack: HeapStack::<SiblingInsertionCandidate>::new_with_capacity(1000),
            temp_aabbs: Default::default(),
            collision_pairs: Vec::new(),
            rebuild_method: Default::default(),
        }
    }
}

impl PhysicsWorld {
    pub fn add_sphere(&mut self, position: Vec3A, velocity: Vec3A, radius: f32, mass: f32) -> u32 {
        self.items.push(SphereCollider {
            position,
            velocity,
            radius,
            mass,
            oversized_aabb: Aabb::empty(),
            min_aabb: Aabb::empty(),
        });
        self.items.len() as u32 - 1
    }

    pub fn add_sphere_update_bvh(
        &mut self,
        position: Vec3A,
        velocity: Vec3A,
        radius: f32,
        mass: f32,
    ) -> u32 {
        let id = self.add_sphere(position, velocity, radius, mass);
        let aabb = match self.rebuild_method {
            BvhRebuid::Full => self.items[id as usize].min_aabb,
            BvhRebuid::Reinsert | BvhRebuid::RemoveAndInsert => {
                self.items[id as usize].oversized_aabb
            }
        };
        self.bvh
            .insert_primitive(aabb, id, &mut self.bvh_insertion_stack);
        id
    }

    pub fn bvh_full_rebuild(&mut self) {
        let oversize_factor = self.oversize_factor();
        self.temp_aabbs.clear();
        let indices = (0..self.items.len() as u32).collect::<Vec<_>>();
        for item in &mut self.items {
            item.update_minimum_aabb();
            item.update_oversized_aabb(oversize_factor);
            if self.rebuild_method == BvhRebuid::Full {
                self.temp_aabbs.push(item.min_aabb);
            } else {
                self.temp_aabbs.push(item.oversized_aabb);
            }
        }
        self.bvh =
            PlocSearchDistance::VeryLow.build(&self.temp_aabbs, indices, SortPrecision::U64, 1);
    }

    pub fn bvh_partial_rebuild_remove_insert(&mut self) {
        let oversize_factor = self.oversize_factor();
        for (primitive_id, item) in self.items.iter_mut().enumerate() {
            if item.update_oversized_aabb(oversize_factor) {
                self.bvh.remove_primitive(primitive_id as u32);
                self.bvh.insert_primitive(
                    item.oversized_aabb,
                    primitive_id as u32,
                    &mut self.bvh_insertion_stack,
                );
            }
        }
    }

    pub fn bvh_partial_rebuild_reinsert(&mut self) {
        let oversize_factor = self.oversize_factor();
        self.bvh.init_primitives_to_nodes();
        let mut stack = HeapStack::new_with_capacity(2000);
        for (primitive_id, item) in self.items.iter_mut().enumerate() {
            if item.update_oversized_aabb(oversize_factor) {
                let node_id = self.bvh.primitives_to_nodes[primitive_id];
                self.bvh.resize_node(node_id as usize, item.oversized_aabb);
                self.bvh.reinsert_node(node_id as usize, &mut stack);
            }
        }
    }

    pub fn oversize_factor(&self) -> f32 {
        match self.rebuild_method {
            BvhRebuid::Full => 0.0,
            BvhRebuid::Reinsert => 0.0,
            BvhRebuid::RemoveAndInsert => AABB_OVERSIZE_FACTOR,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SphereCollider {
    oversized_aabb: Aabb,
    min_aabb: Aabb,
    position: Vec3A,
    velocity: Vec3A,
    radius: f32,
    mass: f32,
}

impl SphereCollider {
    fn update_oversized_aabb(&mut self, oversize_factor: f32) -> bool {
        // Check if any part of the minimum aabb is outside the oversized aabb
        let need_to_update = self.oversized_aabb.min.min(self.min_aabb.min)
            != self.oversized_aabb.min
            || self.oversized_aabb.max.max(self.min_aabb.max) != self.oversized_aabb.max;

        if need_to_update {
            self.oversized_aabb = Aabb::new(
                self.min_aabb.min - self.radius * oversize_factor,
                self.min_aabb.max + self.radius * oversize_factor,
            );
        }

        need_to_update
    }

    fn update_minimum_aabb(&mut self) {
        self.min_aabb = Aabb::new(self.position - self.radius, self.position + self.radius);
    }
}

fn physics_update(physics: &mut PhysicsWorld) {
    let gravity = vec3a(0.0, -9.81, 0.0);
    let box_size = 15.0;
    let wall_damping = 0.5;
    let sphere_damping = 0.8;

    {
        scope!("gravity and walls");
        for sphere in &mut physics.items {
            sphere.velocity += gravity * DT;
            sphere.position += sphere.velocity * DT;

            // Ground collision
            if sphere.position.y - sphere.radius < 0.0 {
                sphere.position.y = sphere.radius;
                sphere.velocity.y *= -wall_damping;
            }

            // Walls collision
            for axis in [0, 2] {
                if sphere.position[axis] - sphere.radius < -box_size {
                    sphere.position[axis] = sphere.radius - box_size;
                    sphere.velocity[axis] *= -wall_damping;
                }
                if sphere.position[axis] + sphere.radius > box_size {
                    sphere.position[axis] = box_size - sphere.radius;
                    sphere.velocity[axis] *= -wall_damping;
                }
            }
            sphere.update_minimum_aabb();
        }
    }

    if !NO_BVH {
        scope!("rebuild");
        match physics.rebuild_method {
            BvhRebuid::Full => physics.bvh_full_rebuild(),
            BvhRebuid::Reinsert => physics.bvh_partial_rebuild_reinsert(),
            BvhRebuid::RemoveAndInsert => physics.bvh_partial_rebuild_remove_insert(),
        }
    }

    physics.collision_pairs.clear();

    if NO_BVH {
        scope!("find collision pairs, brute force");
        let len = physics.items.len();
        for s1 in 0..len {
            let s1_min_aabb = physics.items[s1].min_aabb;
            for s2 in (s1 + 1)..len {
                if physics.items[s2].min_aabb.intersect_aabb(&s1_min_aabb) && s1 != s2 {
                    physics
                        .collision_pairs
                        .push(Pair::new(s1 as u32, s2 as u32));
                }
            }
        }
    } else {
        scope!("find collision pairs, bvh");
        let mut traversal_stack = HeapStack::new_with_capacity(2000);
        for s1 in 0..physics.items.len() {
            let s1_min_aabb = physics.items[s1].min_aabb;

            // Split borrows (WHYYYYYYYYYYYYYYYYYYYYY)
            let (pairs, bvh, items) = (
                &mut physics.collision_pairs,
                &physics.bvh,
                &mut physics.items,
            );

            bvh.aabb_traverse(&mut traversal_stack, s1_min_aabb, |bvh, node_id| {
                let node = &bvh.nodes[node_id as usize];
                let start = node.first_index as usize;
                let end = start + node.prim_count as usize;
                for node_prim_id in start..end {
                    let s2 = bvh.primitive_indices[node_prim_id] as usize;
                    // Check against all primitives in this leaf node
                    if items[s2].min_aabb.intersect_aabb(&s1_min_aabb) && s1 != s2 {
                        pairs.push(Pair::new(s1 as u32, s2 as u32));
                    }
                }
                true
            });
        }
    }

    {
        scope!("sort pairs and dedup");
        physics.collision_pairs.sort_unstable(); // unstable should be still deterministic in this case since no dup
        physics.collision_pairs.dedup(); // dedup not needed with brute force method?
    }

    if VERIFY_PAIRS {
        verify_pairs(&physics);
    }

    {
        scope!("resolve collisions");
        // Split borrows (WHYYYYYYYYYYYYYYYYYYYYY)
        let (pairs, mut items) = (&physics.collision_pairs, &mut physics.items);
        for pair in pairs {
            let (s1, s2) = pair.get();
            resolve_collision(&mut items, s1 as usize, s2 as usize, sphere_damping);
        }
    }
}

fn verify_pairs(physics: &PhysicsWorld) {
    let mut temp_pairs = Vec::with_capacity(physics.collision_pairs.len() * 2);
    let len = physics.items.len();
    for s1 in 0..len {
        let s1_min_aabb = physics.items[s1].min_aabb;
        for s2 in (s1 + 1)..len {
            if physics.items[s2].min_aabb.intersect_aabb(&s1_min_aabb) && s1 != s2 {
                temp_pairs.push(Pair::new(s1 as u32, s2 as u32));
            }
        }
    }
    temp_pairs.sort_unstable();
    temp_pairs.dedup();
    assert_eq!(temp_pairs.len(), physics.collision_pairs.len());
}

fn resolve_collision(spheres: &mut [SphereCollider], s1: usize, s2: usize, damping: f32) -> bool {
    let delta = spheres[s2].position - spheres[s1].position;
    let dist_sq = delta.length_squared();
    let radius_sum = spheres[s1].radius + spheres[s2].radius;
    let radius_sum_sq = radius_sum * radius_sum;

    if dist_sq < radius_sum_sq && dist_sq > 0.0 {
        let dist = dist_sq.sqrt();
        let normal = delta / dist;

        let penetration_depth = radius_sum - dist;
        let correction = normal * (penetration_depth / 2.0);

        spheres[s1].position -= correction;
        spheres[s2].position += correction;

        let relative_velocity = spheres[s2].velocity - spheres[s1].velocity;
        let velocity_along_normal = relative_velocity.dot(normal);

        if velocity_along_normal > 0.0 {
            return false;
        }

        let impulse = -(1.0 + damping) * velocity_along_normal
            / (1.0 / spheres[s1].mass + 1.0 / spheres[s2].mass);
        let impulse_vector = normal * impulse;

        spheres[s1].velocity -= impulse_vector / spheres[s1].mass;
        spheres[s2].velocity += impulse_vector / spheres[s2].mass;
        true
    } else {
        false
    }
}

// https://iquilezles.org/articles/intersectors/
fn ray_sphere_intersect(ray: &Ray, center: Vec3A, radius: f32) -> f32 {
    let oc = ray.origin - center;
    let b = oc.dot(ray.direction);
    let qc = oc - b * ray.direction;
    let mut h = radius * radius - qc.dot(qc);
    if h < 0.0 {
        // no intersection
        return f32::INFINITY;
    };
    h = h.sqrt();
    let t = vec2(-b - h, -b + h);

    if t.y < 0.0 {
        f32::INFINITY // ray does NOT intersect the sphere
    } else if t.x < 0.0 {
        t.y // origin inside the sphere, t.y is intersection distance
    } else {
        // origin outside the sphere, t.x is intersection distance
        t.x
    }
}
