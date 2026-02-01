// Uses cost / merging from cwbvh paper

use glam::{UVec3, Vec3A, vec3a};

use crate::{
    PerComponent,
    aabb::Aabb,
    bvh2::Bvh2,
    cwbvh::{BRANCHING, CwBvh, CwBvhNode, DENOM},
};

use super::DIRECTIONS;

/// Convert a bvh2 to CwBvh
pub struct Bvh2Converter<'a> {
    pub bvh2: &'a Bvh2,
    pub nodes: Vec<CwBvhNode>,
    pub primitive_indices: Vec<u32>,
    pub decisions: Vec<Decision>,
    pub order_children_during_build: bool,
    pub include_exact_node_aabbs: bool,
    pub exact_node_aabbs: Option<Vec<Aabb>>,
    direction_lut: [Vec3A; 8],
}

const INVALID: u8 = u8::MAX;
const INVALID32: u32 = u32::MAX;
const INVALID_USIZE: usize = INVALID32 as usize;

const PRIM_COST: f32 = 0.3;

impl<'a> Bvh2Converter<'a> {
    /// Initialize the Bvh2 to CwBvh converter.
    pub fn new(bvh2: &'a Bvh2, order_children: bool, include_exact_node_aabbs: bool) -> Self {
        let capacity = bvh2.primitive_indices.len();

        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(Default::default());

        let mut direction_lut = [Vec3A::ZERO; DIRECTIONS];
        direction_lut
            .iter_mut()
            .enumerate()
            .for_each(|(s, direction)| {
                *direction = vec3a(
                    if (s & 0b100) != 0 { -1.0 } else { 1.0 },
                    if (s & 0b010) != 0 { -1.0 } else { 1.0 },
                    if (s & 0b001) != 0 { -1.0 } else { 1.0 },
                );
            });

        Self {
            bvh2,
            nodes,
            primitive_indices: Vec::with_capacity(capacity),
            decisions: vec![Decision::default(); bvh2.nodes.len() * 7],
            order_children_during_build: order_children,
            direction_lut,
            include_exact_node_aabbs,
            exact_node_aabbs: if include_exact_node_aabbs {
                Some(vec![Aabb::empty(); bvh2.nodes.len()])
            } else {
                None
            },
        }
    }

    /// Convert the bvh2 to CwBvh
    pub fn convert_to_cwbvh(&mut self) {
        crate::scope!("convert_to_cwbvh");
        debug_assert_eq!(std::mem::size_of::<CwBvhNode>(), 80);
        self.convert_to_cwbvh_impl(0, 0);
    }

    pub fn convert_to_cwbvh_impl(&mut self, node_index_bvh8: usize, node_index_bvh2: usize) {
        let mut node = self.nodes[node_index_bvh8];
        let aabb = self.bvh2.nodes[node_index_bvh2].aabb();
        if let Some(exact_node_aabbs) = &mut self.exact_node_aabbs {
            exact_node_aabbs[node_index_bvh8] = *aabb;
        }

        let node_p = aabb.min;
        node.p = node_p.into();

        let e = ((aabb.max - aabb.min).max(Vec3A::splat(1e-20)) * DENOM)
            .log2()
            .ceil()
            .exp2();
        debug_assert!(e.cmpgt(Vec3A::ZERO).all(), "aabb: {aabb:?} e: {e}");

        let rcp_e = 1.0 / e;
        let e: UVec3 = e.per_comp(|c: f32| {
            let bits = c.to_bits();
            // Only the exponent bits can be non-zero
            debug_assert_eq!(bits & 0b10000000011111111111111111111111, 0);
            bits >> 23
        });
        node.e = [e.x as u8, e.y as u8, e.z as u8];

        let children = &mut [INVALID32; 8];

        let child_count = &mut 0;
        self.get_children(node_index_bvh2, children, child_count, 0);

        if self.order_children_during_build {
            self.order_children(node_index_bvh2, children, *child_count as usize);
        }

        node.imask = 0;

        node.primitive_base_idx = self.primitive_indices.len() as u32;
        node.child_base_idx = self.nodes.len() as u32;

        let mut num_internal_nodes = 0;
        let mut num_primitives = 0_u32;

        for (i, child_index) in children.iter().enumerate() {
            if *child_index == INVALID32 {
                continue; // Empty slot
            };

            let child_aabb = self.bvh2.nodes[*child_index as usize].aabb();

            // const PAD: f32 = 1e-20;
            // Use to force non-zero volumes.
            const PAD: f32 = 0.0;

            let mut child_min = ((child_aabb.min - node_p - PAD) * rcp_e).floor();
            let mut child_max = ((child_aabb.max - node_p + PAD) * rcp_e).ceil();

            child_min = child_min.clamp(Vec3A::ZERO, Vec3A::splat(255.0));
            child_max = child_max.clamp(Vec3A::ZERO, Vec3A::splat(255.0));

            debug_assert!((child_min.cmple(child_max)).all());

            node.child_min_x[i] = child_min.x as u8;
            node.child_min_y[i] = child_min.y as u8;
            node.child_min_z[i] = child_min.z as u8;
            node.child_max_x[i] = child_max.x as u8;
            node.child_max_y[i] = child_max.y as u8;
            node.child_max_z[i] = child_max.z as u8;

            match self.decisions[(child_index * 7) as usize].kind {
                DecisionKind::LEAF => {
                    let primitive_count = self.count_primitives(*child_index as usize, self.bvh2);
                    debug_assert!(primitive_count > 0 && primitive_count <= 3);

                    // Three highest bits contain unary representation of primitive count

                    node.child_meta[i] = num_primitives as u8
                        | match primitive_count {
                            1 => 0b0010_0000,
                            2 => 0b0110_0000,
                            3 => 0b1110_0000,
                            _ => panic!("Incorrect leaf primitive count: {primitive_count}"),
                        };

                    num_primitives += primitive_count;
                    debug_assert!(num_primitives <= 24);
                }
                DecisionKind::INTERNAL => {
                    node.imask |= 1u8 << i;

                    node.child_meta[i] = (24 + i as u8) | 0b0010_0000;

                    num_internal_nodes += 1;
                }
                DecisionKind::DISTRIBUTE => unreachable!(),
            }
        }

        self.nodes
            .resize(self.nodes.len() + num_internal_nodes, Default::default());
        self.nodes[node_index_bvh8] = node;

        debug_assert!(node.child_base_idx as usize + num_internal_nodes == self.nodes.len());
        debug_assert!(
            node.primitive_base_idx + num_primitives == self.primitive_indices.len() as u32
        );

        // Recurse on Internal Nodes
        let mut offset = 0;
        for (i, child_index) in children.iter().enumerate() {
            if *child_index != INVALID32 && (node.imask & (1 << i)) != 0 {
                self.convert_to_cwbvh_impl(
                    (node.child_base_idx + offset) as usize,
                    *child_index as usize,
                );
                offset += 1;
            }
        }
        //self.nodes[node_index_bvh8] = node;
    }

    // Recursively count primitives in subtree of the given Node
    // Simultaneously fills the indices buffer of the BVH8
    fn count_primitives(&mut self, node_index: usize, bvh2: &Bvh2) -> u32 {
        let node = bvh2.nodes[node_index];

        if node.is_leaf() {
            debug_assert!(node.prim_count == 1);

            self.primitive_indices
                .push(bvh2.primitive_indices[node.first_index as usize]);

            return node.prim_count;
        }

        self.count_primitives(node.first_index as usize, bvh2)
            + self.count_primitives((node.first_index + 1) as usize, bvh2)
    }

    /// Fill cost table for bvh2 -> bvh8 conversion
    pub fn calculate_cost(&mut self, max_prims_per_leaf: u32) {
        crate::scope!("calculate_cost");
        self.calculate_cost_impl(0, max_prims_per_leaf, 0);
    }

    // Based on https://github.com/jan-van-bergen/GPU-Raytracer/blob/6559ae2241c8fdea0ddaec959fe1a47ec9b3ab0d/Src/BVH/Converters/BVH8Converter.cpp#L24
    pub fn calculate_cost_impl(
        &mut self,
        node_index: usize,
        max_prims_per_leaf: u32,
        _current_depth: i32,
    ) -> u32 {
        let node = &self.bvh2.nodes[node_index];
        let half_area = node.aabb().half_area();
        let first_index = node.first_index;
        let prim_count = node.prim_count;

        let node_dec_idx = node_index * 7;
        let first_index_7 = (first_index * 7) as usize;
        let next_index_7 = ((first_index + 1) * 7) as usize;

        let num_primitives;

        // TODO possibly merge as much as possible past a specified depth
        // let depth_cost = if current_depth > 15 { 1.0 } else { 1.0 };

        //if is_leaf()
        if prim_count != 0 {
            num_primitives = prim_count;
            if num_primitives != 1 {
                panic!(
                    "ERROR: BVH8 Builder expects BVH with leaf Nodes containing only 1 primitive!\n"
                );
            }

            // SAH cost
            let cost_leaf = half_area * (num_primitives as f32) * PRIM_COST;

            for i in 0..7 {
                let decision = &mut self.decisions[node_dec_idx + i];
                decision.kind = DecisionKind::LEAF;
                decision.cost = cost_leaf;
            }
        } else {
            num_primitives = self.calculate_cost_impl(
                first_index as usize,
                max_prims_per_leaf,
                _current_depth + 1,
            ) + self.calculate_cost_impl(
                (first_index + 1) as usize,
                max_prims_per_leaf,
                _current_depth + 1,
            );

            // Separate case: i=0 (i=1 in the paper)
            {
                let cost_leaf = if num_primitives <= max_prims_per_leaf {
                    (num_primitives as f32) * half_area * PRIM_COST
                } else {
                    f32::INFINITY
                };

                let mut cost_distribute = f32::INFINITY;

                let mut distribute_left = INVALID;
                let mut distribute_right = INVALID;

                for k in 0..7 {
                    let c = self.decisions[first_index_7 + k].cost
                        + self.decisions[next_index_7 + 6 - k].cost;

                    if c < cost_distribute {
                        cost_distribute = c;

                        distribute_left = k as u8;
                        distribute_right = 6 - k as u8;
                    }
                }

                let cost_internal = cost_distribute + half_area;

                let decision = &mut self.decisions[node_dec_idx];
                if cost_leaf < cost_internal {
                    decision.kind = DecisionKind::LEAF;
                    decision.cost = cost_leaf;
                } else {
                    decision.kind = DecisionKind::INTERNAL;
                    decision.cost = cost_internal;
                }

                decision.distribute_left = distribute_left;
                decision.distribute_right = distribute_right;
            }

            // In the paper i=2..7
            let mut node_i;
            for i in 1..7 {
                node_i = node_dec_idx + i;
                let mut cost_distribute = self.decisions[node_i - 1].cost;

                let mut distribute_left = INVALID;
                let mut distribute_right = INVALID;

                for k in 0..i {
                    let c = self.decisions[first_index_7 + k].cost
                        + self.decisions[next_index_7 + i - k - 1].cost;

                    if c < cost_distribute {
                        cost_distribute = c;

                        let k_u8 = k as u8;
                        distribute_left = k_u8;
                        distribute_right = i as u8 - k_u8 - 1;
                    }
                }

                let decision = &mut self.decisions[node_i];
                decision.cost = cost_distribute;

                if distribute_left != INVALID {
                    decision.kind = DecisionKind::DISTRIBUTE;
                    decision.distribute_left = distribute_left;
                    decision.distribute_right = distribute_right;
                } else {
                    self.decisions[node_i] = self.decisions[node_i - 1];
                }
            }
        }

        num_primitives
    }

    pub fn get_children(
        &mut self,
        node_index: usize,
        children: &mut [u32; 8],
        child_count: &mut u32,
        i: usize,
    ) {
        let node = &self.bvh2.nodes[node_index];

        if node.is_leaf() {
            children[*child_count as usize] = node_index as u32;
            *child_count += 1;
            return;
        }

        let decision = &self.decisions[node_index * 7 + i];
        let distribute_left = decision.distribute_left;
        let distribute_right = decision.distribute_right;

        debug_assert!(distribute_left < 7);
        debug_assert!(distribute_right < 7);

        // Recurse on left child if it needs to distribute
        if self.decisions[(node.first_index * 7 + distribute_left as u32) as usize].kind
            == DecisionKind::DISTRIBUTE
        {
            self.get_children(
                node.first_index as usize,
                children,
                child_count,
                distribute_left as usize,
            );
        } else {
            children[*child_count as usize] = node.first_index;
            *child_count += 1;
        }

        // Recurse on right child if it needs to distribute
        if self.decisions[((node.first_index + 1) * 7 + distribute_right as u32) as usize].kind
            == DecisionKind::DISTRIBUTE
        {
            self.get_children(
                (node.first_index + 1) as usize,
                children,
                child_count,
                distribute_right as usize,
            );
        } else {
            children[*child_count as usize] = node.first_index + 1;
            *child_count += 1;
        }
    }

    /// Arrange child nodes in Morton order according to their centroids so that the order in which the intersected
    /// children are traversed can be determined by the ray octant.
    // Based on https://github.com/jan-van-bergen/GPU-Raytracer/blob/6559ae2241c8fdea0ddaec959fe1a47ec9b3ab0d/Src/BVH/Converters/BVH8Converter.cpp#L148
    pub fn order_children(
        &mut self,
        node_index: usize,
        children: &mut [u32; 8],
        child_count: usize,
    ) {
        let node = &self.bvh2.nodes[node_index];
        let p = node.aabb().center();

        let mut cost = [[f32::MAX; DIRECTIONS]; BRANCHING];

        assert!(child_count <= BRANCHING);
        assert!(cost.len() >= child_count);
        // Fill cost table
        for s in 0..DIRECTIONS {
            let d = self.direction_lut[s];
            for c in 0..child_count {
                let v = self.bvh2.nodes[children[c] as usize].aabb().center() - p;
                let cost_slot = unsafe { cost.get_unchecked_mut(c).get_unchecked_mut(s) };
                *cost_slot = d.dot(v); // No benefit from normalizing
            }
        }

        let mut assignment = [INVALID_USIZE; BRANCHING];
        let mut slot_filled = [false; DIRECTIONS];

        // The paper suggests the auction method, but greedy is almost as good.
        loop {
            let mut min_cost = f32::MAX;

            let mut min_slot = INVALID_USIZE;
            let mut min_index = INVALID_USIZE;

            // Find cheapest unfilled slot of any unassigned child
            for c in 0..child_count {
                if assignment[c] == INVALID_USIZE {
                    for (s, &slot_filled) in slot_filled.iter().enumerate() {
                        let cost = unsafe { *cost.get_unchecked(c).get_unchecked(s) };
                        if !slot_filled && cost < min_cost {
                            min_cost = cost;

                            min_slot = s;
                            min_index = c;
                        }
                    }
                }
            }

            if min_slot == INVALID_USIZE {
                break;
            }

            slot_filled[min_slot] = true;
            assignment[min_index] = min_slot;
        }

        let original_order = std::mem::replace(children, [INVALID32; 8]);

        assert!(assignment.len() >= child_count); // Allow compiler to skip bounds check
        assert!(original_order.len() >= child_count); // Allow compiler to skip bounds check
        for i in 0..child_count {
            debug_assert!(assignment[i] != INVALID_USIZE);
            debug_assert!(original_order[i] != INVALID32);
            children[assignment[i]] = original_order[i];
        }
    }
}

#[derive(Copy, Clone, PartialEq, Default)]
pub enum DecisionKind {
    LEAF,
    INTERNAL,
    #[default]
    DISTRIBUTE,
}

#[derive(Copy, Clone, Default)]
pub struct Decision {
    pub cost: f32,
    pub kind: DecisionKind,
    pub distribute_left: u8,
    pub distribute_right: u8,
}

/// Convert the given bvh2 to cwbvh
/// # Arguments
/// * `bvh2` - Source BVH
/// * `max_prims_per_leaf` - 0..=3 The maximum number of primitives per leaf.
pub fn bvh2_to_cwbvh(
    bvh2: &Bvh2,
    max_prims_per_leaf: u32,
    order_children: bool,
    include_exact_node_aabbs: bool,
) -> CwBvh {
    if bvh2.nodes.is_empty() {
        return CwBvh::default();
    }
    let mut converter = Bvh2Converter::new(bvh2, order_children, include_exact_node_aabbs);
    converter.calculate_cost(max_prims_per_leaf);
    converter.convert_to_cwbvh();

    CwBvh {
        nodes: converter.nodes,
        primitive_indices: converter.primitive_indices,
        total_aabb: *bvh2.nodes[0].aabb(),
        exact_node_aabbs: converter.exact_node_aabbs,
        uses_spatial_splits: bvh2.uses_spatial_splits,
    }
}
