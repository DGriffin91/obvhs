use bytemuck::{Pod, Zeroable};
#[cfg(feature = "small_bvh2_node")]
use glam::Vec3;

use crate::aabb::Aabb;

#[cfg(feature = "small_bvh2_node")]
static_assertions::assert_eq_size!(Bvh2Node, Aabb);
#[cfg(feature = "small_bvh2_node")]
static_assertions::assert_eq_align!(Bvh2Node, Aabb);

#[cfg(feature = "small_bvh2_node")]
/// A node in the Bvh2, can be an inner node or leaf.
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
#[repr(align(16))]
pub struct Bvh2Node {
    /// The bounding box minimum coordinate for the primitive(s) contained in this node
    pub min: Vec3,
    /// Number of primitives contained in this node.
    /// If prim_count is 0, this is a inner node.
    /// If prim_count > 0 this node is a leaf node.
    pub prim_count: u32,
    /// The bounding box maximum coordinate for the primitive(s) contained in this node
    pub max: Vec3,
    /// The index of the first child Aabb or primitive.
    /// If this node is an inner node the first child will be at `nodes[first_index]`, and the second at `nodes[first_index + 1]`.
    /// If this node is a leaf node the first index typically indexes into a primitive_indices list that contains the actual index of the primitive.
    /// The reason for this mapping is that if multiple primitives are contained in this node, they need to have their indices layed out contiguously.
    /// To avoid this indirection we have two options:
    /// 1. Layout the primitives in the order of the primitive_indices mapping so that this can index directly into the primitive list.
    /// 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    pub first_index: u32,
}

#[cfg(not(feature = "small_bvh2_node"))]
/// A node in the Bvh2, can be an inner node or leaf.
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
#[repr(align(16))]
pub struct Bvh2Node {
    /// The bounding box for the primitive(s) contained in this node
    pub aabb: Aabb,
    /// Number of primitives contained in this node.
    /// If prim_count is 0, this is a inner node.
    /// If prim_count > 0 this node is a leaf node.
    pub prim_count: u32,
    /// The index of the first child Aabb or primitive.
    /// If this node is an inner node the first child will be at `nodes[first_index]`, and the second at `nodes[first_index + 1]`.
    /// If this node is a leaf node the first index typically indexes into a primitive_indices list that contains the actual index of the primitive.
    /// The reason for this mapping is that if multiple primitives are contained in this node, they need to have their indices layed out contiguously.
    /// To avoid this indirection we have two options:
    /// 1. Layout the primitives in the order of the primitive_indices mapping so that this can index directly into the primitive list.
    /// 2. Only allow one primitive per node and write back the original mapping to the bvh node list.
    pub first_index: u32,
    /// With the aabb, prim_count, and first_index, this struct was already padded out to 48 bytes. These meta fields
    /// allow the user to access this otherwise unused space.
    pub meta1: u32,
    /// With the aabb, prim_count, and first_index, this struct was already padded out to 48 bytes. These meta fields
    /// allow the user to access this otherwise unused space.
    pub meta2: u32,
}

impl Bvh2Node {
    #[cfg(feature = "small_bvh2_node")]
    #[inline(always)]
    pub fn new(aabb: Aabb, prim_count: u32, first_index: u32) -> Self {
        let mut node: Bvh2Node = bytemuck::cast(aabb);
        node.prim_count = prim_count;
        node.first_index = first_index;
        node
    }

    #[cfg(not(feature = "small_bvh2_node"))]
    #[inline(always)]
    pub fn new(aabb: Aabb, prim_count: u32, first_index: u32) -> Self {
        Bvh2Node {
            aabb,
            prim_count,
            first_index,
            meta1: Default::default(),
            meta2: Default::default(),
        }
    }

    #[cfg(feature = "small_bvh2_node")]
    #[inline(always)]
    pub fn aabb(&self) -> &Aabb {
        bytemuck::cast_ref(self)
    }

    #[cfg(not(feature = "small_bvh2_node"))]
    #[inline(always)]
    pub fn aabb(&self) -> &Aabb {
        &self.aabb
    }

    #[cfg(feature = "small_bvh2_node")]
    #[inline(always)]
    pub fn set_aabb(&mut self, aabb: Aabb) {
        self.min = aabb.min.into();
        self.max = aabb.max.into();
    }

    #[cfg(not(feature = "small_bvh2_node"))]
    #[inline(always)]
    pub fn set_aabb(&mut self, aabb: Aabb) {
        self.aabb = aabb;
    }

    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.prim_count != 0
    }

    #[inline(always)]
    pub fn is_left_sibling(node_id: usize) -> bool {
        node_id % 2 == 1
    }
    #[inline(always)]
    pub fn get_sibling_id(node_id: usize) -> usize {
        if Self::is_left_sibling(node_id) {
            node_id + 1
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_left_sibling_id(node_id: usize) -> usize {
        if Self::is_left_sibling(node_id) {
            node_id
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_right_sibling_id(node_id: usize) -> usize {
        if Self::is_left_sibling(node_id) {
            node_id + 1
        } else {
            node_id
        }
    }

    #[inline(always)]
    pub fn is_left_sibling32(node_id: u32) -> bool {
        node_id % 2 == 1
    }
    #[inline(always)]
    pub fn get_sibling_id32(node_id: u32) -> u32 {
        if Self::is_left_sibling32(node_id) {
            node_id + 1
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_left_sibling_id32(node_id: u32) -> u32 {
        if Self::is_left_sibling32(node_id) {
            node_id
        } else {
            node_id - 1
        }
    }
    #[inline(always)]
    pub fn get_right_sibling_id32(node_id: u32) -> u32 {
        if Self::is_left_sibling32(node_id) {
            node_id + 1
        } else {
            node_id
        }
    }
    #[inline(always)]
    pub fn make_inner(&mut self, first_index: u32) {
        self.prim_count = 0;
        self.first_index = first_index;
    }
}
