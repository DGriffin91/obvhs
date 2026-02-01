use core::f32;
use std::{
    str::FromStr,
    time::{Duration, Instant},
};

use argh::FromArgs;
use glam::*;
use minifb::{Key, MouseButton, Window, WindowOptions};
use obvhs::{
    PrettyDuration,
    aabb::Aabb,
    bvh2::{Bvh2, insertion_removal::SiblingInsertionCandidate, reinsertion::ReinsertionOptimizer},
    faststack::HeapStack,
    ploc::{PlocBuilder, PlocSearchDistance, SortPrecision, rebuild::compute_rebuild_path_flags},
    ray::{Ray, RayHit},
};

#[cfg(feature = "parallel")]
use rayon::iter::IntoParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
#[cfg(feature = "parallel")]
use std::cell::RefCell;
#[cfg(feature = "parallel")]
use thread_local::ThreadLocal;

#[path = "./helpers/debug.rs"]
mod debug;
use debug::{color_to_minifb_pixel, text::Text, timer::reset_timers};

use crate::debug::timer::for_each_sorted_timer;

#[derive(FromArgs, Default)]
/// `physics` example.
/// Runs an extremely simple physics simulation, optionally using a bvh for the broad phase and real time CPU ray
/// tracing. Click the window to drop another ball. Run with `--release` for best performance.
struct Args {
    /// disables the renderer and window. Only runs physics sim for `count` steps, disables ray tracing.
    #[argh(switch)]
    no_render: bool,
    /// disables using the bvh for physics broad phase (still uses it for rendering)
    #[argh(switch)]
    no_physics_bvh: bool,
    /// bvh update method. Modes: 'rebuild', 'reinsert', 'parallel_reinsert', 'remove_and_insert', "partial_rebuild"
    #[argh(option, default = "BvhUpdate::Rebuild")]
    bvh_update: BvhUpdate,
    /// check that we got the same list of pairs from the bvh broad phase as brute force method
    #[argh(switch)]
    verify_pairs: bool,
    /// how much to oversize the AABBs when using Reinsert or RemoveAndInsert `min_aabb + radius * aabb_oversize_factor`
    #[argh(option, default = "0.3")]
    aabb_oversize: f32,
    /// physics delta step. Constant so it's deterministic.
    #[argh(option, default = "0.015")]
    dt: f32,
    /// initial sphere count
    #[argh(option, default = "10000")]
    count: u32,
    /// how many steps to take when the renderer is disabled
    #[argh(option, default = "4000")]
    bench_steps: u32,
    /// fps limit when rending output (also limits physics sim), 0 for none
    #[argh(option, default = "120")]
    fps_limit: usize,
    /// render resolution (width:height are 1:1)
    #[argh(option, default = "512")]
    render_res: usize,
    /// render on a single thread (control building parallelism with `--features parallel`)
    #[argh(switch)]
    single_threaded_render: bool,
}

fn main() {
    let config: Args = argh::from_env();
    let mut physics = PhysicsWorld {
        config,
        ..Default::default()
    };

    // Setup render target and camera
    let width = physics.config.render_res;
    let height = physics.config.render_res;
    let debug_renderer = DebugRenderer::new(width, height);

    for i in 0..physics.config.count {
        let x = (i as f32 * 0.001).sin() * 1.0;
        let z = (i as f32 * 0.001).cos() * 1.0;
        physics.add_sphere(vec3a(x, i as f32 * 0.25, z), Vec3A::ZERO, 0.3, 0.5);
    }
    physics.bvh_full_rebuild();

    if physics.config.no_render {
        println!("Initial bvh depth: {}", physics.bvh.depth(0));
        let start = std::time::Instant::now();
        for _ in 0..physics.config.bench_steps {
            physics_update(&mut physics);
        }
        let elapsed = start.elapsed();
        println!("End bvh depth: {}", physics.bvh.depth(0));
        println!(
            "{:>8} bench | {} per iter",
            format!("{}", PrettyDuration(elapsed)),
            format!(
                "{}",
                PrettyDuration(Duration::from_secs_f32(
                    elapsed.as_secs_f32() / physics.config.bench_steps as f32
                ))
            ),
        );
        println!("-------------------------------");
        for_each_sorted_timer(|text, timer| {
            println!(
                "{:>7}x {:>7} {text}",
                format!("{}", timer.count),
                format!("{}", PrettyDuration(timer.total_duration / timer.count))
            )
        })
    } else {
        let mut text = Text::new(width, height, 1);
        let mut window = Window::new("", width, height, WindowOptions::default()).unwrap();
        window.set_target_fps(physics.config.fps_limit);
        let mut buffer = vec![0u32; width * height];
        let mut spawn = false;
        while window.is_open() && !window.is_key_down(Key::Escape) {
            reset_timers();
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

            let render_start = Instant::now();
            debug_renderer.render(&physics, &mut buffer, &physics.bvh);
            let render_end = render_start.elapsed();

            text.draw_timers(&mut buffer, (4, 4), 10);
            text.dbg_pt(
                &mut buffer,
                &format!(
                    "{}/{}",
                    physics.updated_leaves_this_frame,
                    physics.items.len(),
                ),
                "updated",
            );

            window.update_with_buffer(&buffer, width, height).unwrap();
            window.set_title(&format!(
                "{} physics, {} render",
                PrettyDuration(physics_end),
                PrettyDuration(render_end),
            ));
        }
    }
}

pub struct DebugRenderer {
    pub target_size: Vec2,
    pub eye: Vec3A,
    pub proj_inv: Mat4,
    pub view_inv: Mat4,
    pub width: usize,
    pub height: usize,
}

impl DebugRenderer {
    fn new(width: usize, height: usize) -> Self {
        let target_size = Vec2::new(width as f32, height as f32);
        let fov = 55.0f32;
        let eye = vec3a(30.0, 15.0, 30.0);
        let look_at = vec3(0.0, 9.0, 0.0);
        // Compute camera projection & view matrices
        let aspect_ratio = target_size.x / target_size.y;
        let proj_inv =
            Mat4::perspective_infinite_reverse_rh(fov.to_radians(), aspect_ratio, 0.01).inverse();
        let view_inv = Mat4::look_at_rh(eye.into(), look_at, Vec3::Y).inverse();
        Self {
            target_size,
            eye,
            proj_inv,
            view_inv,
            width,
            height,
        }
    }

    fn render(&self, physics: &PhysicsWorld, buffer: &mut Vec<u32>, bvh: &Bvh2) {
        dbg_scope!("render");

        let render = |(i, color): (usize, &mut u32)| {
            // TODO there seems to be precision issues with rendering if the sphere are far away from the camera.
            // This issue occurs regardless of using a BVH. Maybe something in the projection or ray calculation?
            let frag_coord = uvec2((i % self.width) as u32, (i / self.width) as u32);
            let mut screen_uv = frag_coord.as_vec2() / self.target_size;
            screen_uv.y = 1.0 - screen_uv.y;
            let ndc = screen_uv * 2.0 - Vec2::ONE;
            let clip_pos = vec4(ndc.x, ndc.y, 1.0, 1.0);

            let mut vs_pos = self.proj_inv * clip_pos;
            vs_pos /= vs_pos.w;
            let direction = (Vec3A::from((self.view_inv * vs_pos).xyz()) - self.eye).normalize();
            let ray = Ray::new(self.eye, direction, 0.0, f32::MAX);

            let mut hit = RayHit::none();
            if bvh.ray_traverse(ray, &mut hit, |ray, id| {
                let primitive_id = bvh.primitive_indices[id] as usize;
                let sphere = &physics.items[primitive_id];
                ray_sphere_intersect(ray, sphere.position, sphere.radius)
            }) {
                let primitive_id = bvh.primitive_indices[hit.primitive_id as usize];
                let sphere = &physics.items[primitive_id as usize];
                let hit_p = ray.origin + ray.direction * hit.t;
                let normal = (hit_p - sphere.position).normalize_or_zero();
                *color = color_to_minifb_pixel(normal.extend(1.0));
            } else {
                *color = 0; // Clear
            }
        };

        if physics.config.single_threaded_render {
            buffer.iter_mut().enumerate().for_each(render);
        } else {
            buffer.par_iter_mut().enumerate().for_each(render);
        };
    }
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
enum BvhUpdate {
    #[default]
    Rebuild,
    Reinsert,
    ParallelReinsert,
    RemoveAndInsert,
    PartialRebuild,
}

impl FromStr for BvhUpdate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rebuild" => Ok(Self::Rebuild),
            "reinsert" => Ok(Self::Reinsert),
            "parallel_reinsert" => Ok(Self::ParallelReinsert),
            "remove_and_insert" => Ok(Self::RemoveAndInsert),
            "partial_rebuild" => Ok(Self::PartialRebuild),
            _ => Err(format!(
                "Unknown mode: '{s}', valid modes: 'rebuild', 'reinsert', 'remove_and_insert', 'partial_rebuild'"
            )),
        }
    }
}

struct PhysicsWorld {
    items: Vec<SphereCollider>,
    bvh: Bvh2,
    ploc_builder: PlocBuilder,
    bvh_insertion_stack: HeapStack<SiblingInsertionCandidate>,
    reinsertion_optimizer: ReinsertionOptimizer,
    temp_aabbs: Vec<Aabb>,
    temp_indices: Vec<u32>,
    temp_flags: Vec<bool>,
    collision_pairs: Vec<Pair>,
    #[cfg(feature = "parallel")]
    temp_pairs: ThreadLocal<RefCell<Vec<Pair>>>,
    config: Args,
    updated_leaves_this_frame: usize,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self {
            items: Default::default(),
            bvh: Default::default(),
            ploc_builder: Default::default(),
            reinsertion_optimizer: Default::default(),
            bvh_insertion_stack: HeapStack::<SiblingInsertionCandidate>::new_with_capacity(10000),
            temp_aabbs: Default::default(),
            temp_indices: Default::default(),
            temp_flags: Default::default(),
            collision_pairs: Vec::new(),
            #[cfg(feature = "parallel")]
            temp_pairs: ThreadLocal::new(),
            config: Args::default(),
            updated_leaves_this_frame: Default::default(),
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
        let aabb = match self.config.bvh_update {
            BvhUpdate::Rebuild => self.items[id as usize].min_aabb,
            BvhUpdate::Reinsert
            | BvhUpdate::ParallelReinsert
            | BvhUpdate::RemoveAndInsert
            | BvhUpdate::PartialRebuild => self.items[id as usize].oversized_aabb,
        };
        self.bvh
            .insert_primitive(aabb, id, &mut self.bvh_insertion_stack);
        id
    }

    pub fn bvh_full_rebuild(&mut self) {
        dbg_scope!("bvh_full_rebuild");
        let oversize_factor = self.oversize_factor();
        self.temp_aabbs.clear();
        let indices = (0..self.items.len() as u32).collect::<Vec<_>>();
        self.updated_leaves_this_frame = self.items.len();
        for item in &mut self.items {
            item.update_minimum_aabb();
            item.update_oversized_aabb(oversize_factor); // Currently unused in full rebuild
            if self.config.bvh_update == BvhUpdate::Rebuild {
                self.temp_aabbs.push(item.min_aabb);
            } else {
                self.temp_aabbs.push(item.oversized_aabb);
            }
        }
        self.ploc_builder.build_with_bvh(
            &mut self.bvh,
            PlocSearchDistance::Minimum,
            &self.temp_aabbs,
            indices,
            SortPrecision::U64,
            0,
        );
    }

    pub fn bvh_partial_rebuild_reinsert(&mut self) {
        dbg_scope!("bvh_partial_rebuild_reinsert");
        let oversize_factor = self.oversize_factor();
        self.bvh.init_primitives_to_nodes_if_uninit();
        self.updated_leaves_this_frame = 0;
        for (primitive_id, item) in self.items.iter_mut().enumerate() {
            if item.update_oversized_aabb(oversize_factor) {
                let node_id = self.bvh.primitives_to_nodes[primitive_id];
                self.bvh.resize_node(node_id as usize, item.oversized_aabb);
                self.bvh.reinsert_node(node_id as usize);
                self.updated_leaves_this_frame += 1;
            }
        }
    }

    pub fn bvh_partial_rebuild_parallel_reinsert(&mut self) {
        dbg_scope!("bvh_partial_rebuild_parallel_reinsert");
        let oversize_factor = self.oversize_factor();
        self.bvh.init_primitives_to_nodes_if_uninit();
        self.temp_indices.clear();
        for (primitive_id, item) in self.items.iter_mut().enumerate() {
            if item.update_oversized_aabb(oversize_factor) {
                let node_id = self.bvh.primitives_to_nodes[primitive_id];
                self.bvh.resize_node(node_id as usize, item.oversized_aabb);
                self.temp_indices.push(node_id);
                self.updated_leaves_this_frame += 1;
            }
        }
        self.reinsertion_optimizer
            .run_with_candidates(&mut self.bvh, &self.temp_indices, 1);
    }

    pub fn bvh_partial_rebuild_remove_insert(&mut self) {
        dbg_scope!("bvh_partial_rebuild_remove_insert");
        let oversize_factor = self.oversize_factor();
        self.updated_leaves_this_frame = 0;
        for (primitive_id, item) in self.items.iter_mut().enumerate() {
            if item.update_oversized_aabb(oversize_factor) {
                self.bvh.remove_primitive(primitive_id as u32);
                self.bvh.insert_primitive(
                    item.oversized_aabb,
                    primitive_id as u32,
                    &mut self.bvh_insertion_stack,
                );
                self.updated_leaves_this_frame += 1;
            }
        }
    }

    pub fn bvh_partial_rebuild(&mut self) {
        dbg_scope!("bvh_partial_rebuild");
        let oversize_factor = self.oversize_factor();
        self.updated_leaves_this_frame = 0;
        self.temp_indices.clear();

        self.bvh.init_primitives_to_nodes_if_uninit();
        for (primitive_id, item) in self.items.iter_mut().enumerate() {
            if item.update_oversized_aabb(oversize_factor) {
                let node_id = self.bvh.primitives_to_nodes[primitive_id];
                self.temp_indices.push(node_id);
                self.bvh.nodes[node_id as usize].set_aabb(item.oversized_aabb);
                self.updated_leaves_this_frame += 1;
            }
        }

        self.bvh.init_parents_if_uninit();
        compute_rebuild_path_flags(&self.bvh, &self.temp_indices, &mut self.temp_flags);
        self.ploc_builder.partial_rebuild(
            &mut self.bvh,
            |node_id| self.temp_flags[node_id],
            PlocSearchDistance::Minimum,
            SortPrecision::U64,
            0,
        );
    }

    pub fn oversize_factor(&self) -> f32 {
        match self.config.bvh_update {
            BvhUpdate::Rebuild => 0.0,
            BvhUpdate::Reinsert
            | BvhUpdate::ParallelReinsert
            | BvhUpdate::RemoveAndInsert
            | BvhUpdate::PartialRebuild => self.config.aabb_oversize,
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
        dbg_scope!("gravity and walls");
        for sphere in &mut physics.items {
            sphere.velocity += gravity * physics.config.dt;
            sphere.position += sphere.velocity * physics.config.dt;

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

    match physics.config.bvh_update {
        BvhUpdate::Rebuild => physics.bvh_full_rebuild(),
        BvhUpdate::Reinsert => physics.bvh_partial_rebuild_reinsert(),
        BvhUpdate::ParallelReinsert => physics.bvh_partial_rebuild_parallel_reinsert(),
        BvhUpdate::RemoveAndInsert => physics.bvh_partial_rebuild_remove_insert(),
        BvhUpdate::PartialRebuild => physics.bvh_partial_rebuild(),
    }

    physics.collision_pairs.clear();

    #[cfg(feature = "parallel")]
    {
        for p in physics.temp_pairs.iter_mut() {
            p.borrow_mut().clear();
        }
    }

    if physics.config.no_physics_bvh {
        dbg_scope!("find collision pairs, brute force");
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
        dbg_scope!("find collision pairs, bvh");

        let traverse_closure = |s1| {
            #[cfg(feature = "parallel")]
            let pairs = &mut physics.temp_pairs.get_or_default().borrow_mut();

            #[cfg(not(feature = "parallel"))]
            let pairs = &mut physics.collision_pairs;

            let item: &SphereCollider = &physics.items[s1];
            let s1_min_aabb = item.min_aabb;

            physics.bvh.aabb_traverse(s1_min_aabb, |bvh, node_id| {
                let node = &bvh.nodes[node_id as usize];
                let start = node.first_index as usize;
                let end = start + node.prim_count as usize;
                for node_prim_id in start..end {
                    let s2 = bvh.primitive_indices[node_prim_id] as usize;
                    // Check against all primitives in this leaf node
                    if physics.items[s2].min_aabb.intersect_aabb(&s1_min_aabb) && s1 != s2 {
                        pairs.push(Pair::new(s1 as u32, s2 as u32));
                    }
                }
                true
            });
        };

        let range = 0..physics.items.len();

        #[cfg(feature = "parallel")]
        {
            range.into_par_iter().for_each(traverse_closure);
            for p in physics.temp_pairs.iter_mut() {
                physics.collision_pairs.append(&mut p.borrow_mut());
            }
        }

        #[cfg(not(feature = "parallel"))]
        range.into_iter().for_each(traverse_closure);
    }

    {
        dbg_scope!("sort pairs");
        // unstable should be still deterministic in this case since no dup
        // TODO perf/forte Tried using rdst, was a bit faster without parallel feature, but much slower with parallel feature
        physics.collision_pairs.sort_unstable();
    }

    {
        dbg_scope!("dedup pairs");
        physics.collision_pairs.dedup(); // dedup not needed with brute force method?
    }

    if physics.config.verify_pairs {
        verify_pairs(physics);
    }

    {
        dbg_scope!("resolve collisions");
        // Split borrows (WHYYYYYYYYYYYYYYYYYYYYY)
        let (pairs, items) = (&physics.collision_pairs, &mut physics.items);
        for pair in pairs {
            let (s1, s2) = pair.get();
            // TODO resolve in parallel?
            resolve_collision(items, s1 as usize, s2 as usize, sphere_damping);
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
