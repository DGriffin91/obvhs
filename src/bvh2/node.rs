use bytemuck::Zeroable;

use crate::aabb::Aabb;

pub trait Meta: Copy + Clone + Default + Zeroable {}

#[derive(Default, Clone, Copy, Debug, Zeroable)]
pub struct Pad(pub u64);

impl Meta for Pad {}

/// A node in the Bvh2, can be an inner node or leaf.
#[derive(Default, Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct Bvh2Node<T>
where
    T: Meta,
{
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
    pub meta: T,
}

impl<T> Bvh2Node<T>
where
    T: Meta,
{
    #[inline(always)]
    pub fn new(aabb: Aabb, prim_count: u32, first_index: u32) -> Self {
        Self {
            aabb,
            prim_count,
            first_index,
            meta: Default::default(),
        }
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
