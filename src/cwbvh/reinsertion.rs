use crate::{aabb::Aabb, Boundable};

use super::CwBvh;

pub fn cwbvh_reinsertion<T: Boundable>(cwbvh: &mut CwBvh, direct_layout: bool, primitives: &[T]) {
    let ratio = 2;
    let mut parents = cwbvh.compute_parents();
    let mut touched = vec![false; cwbvh.nodes.len()];
    for a_parent_idx in 1..cwbvh.nodes.len() {
        if a_parent_idx > cwbvh.nodes.len() / ratio {
            dbg!("TODO make sense"); // Also remove this from b
            break;
        }
        let a_parent_aabb = cwbvh.node_aabb(a_parent_idx);
        let a_parent_cost = a_parent_aabb.half_area();
        for a_node_ch in 0..8 {
            let a_parent_minus_this_ch =
                get_total_aabb_minus_one(cwbvh, a_parent_idx, a_node_ch, direct_layout, primitives);
            if cwbvh.nodes[a_parent_idx].is_child_empty(a_node_ch) {
                // TODO allow swapping with empty slots
                continue;
            }
            if cwbvh.nodes[a_parent_idx].is_leaf(a_node_ch) {
                // TODO allow swapping leafs?
                continue;
            }
            let a_child_node_index = cwbvh.nodes[a_parent_idx].child_node_index(a_node_ch) as usize;
            if touched[a_child_node_index] {
                continue;
            }
            let a_child_aabb = cwbvh.node_aabb(a_child_node_index);
            let mut best_swap_parent = 0;
            let mut best_swap_child = 0;
            let mut best_swap_child_inside = 0;
            let mut best_cost_diff = 0.0;
            'b_parent: for b_parent_idx in 1..cwbvh.nodes.len() {
                if b_parent_idx > cwbvh.nodes.len() / ratio {
                    break;
                }
                if b_parent_idx == a_parent_idx {
                    continue;
                }
                let mut hierarchy_test_index = b_parent_idx;
                loop {
                    hierarchy_test_index = parents[hierarchy_test_index] as usize;
                    if hierarchy_test_index == 0 {
                        break;
                    }
                    if hierarchy_test_index == a_child_node_index {
                        continue 'b_parent;
                    }
                }
                let mut hierarchy_test_index = a_parent_idx;
                loop {
                    hierarchy_test_index = parents[hierarchy_test_index] as usize;
                    if hierarchy_test_index == 0 {
                        break;
                    }
                    if hierarchy_test_index == b_parent_idx {
                        continue 'b_parent;
                    }
                }
                let b_parent_aabb = cwbvh.node_aabb(b_parent_idx);
                let b_parent_cost = b_parent_aabb.half_area();
                'b_child: for b_node_ch in 0..8 {
                    let b_parent_minus_this_ch = get_total_aabb_minus_one(
                        cwbvh,
                        b_parent_idx,
                        b_node_ch,
                        direct_layout,
                        primitives,
                    );
                    if cwbvh.nodes[b_parent_idx].is_child_empty(b_node_ch) {
                        // TODO allow swapping with empty slots
                        continue;
                    }
                    if cwbvh.nodes[b_parent_idx].is_leaf(b_node_ch) {
                        // TODO allow swapping leafs?
                        continue;
                    }
                    let b_child_node_index =
                        cwbvh.nodes[b_parent_idx].child_node_index(b_node_ch) as usize;
                    if touched[b_child_node_index] {
                        continue;
                    }

                    let mut hierarchy_test_index = b_child_node_index;
                    loop {
                        hierarchy_test_index = parents[hierarchy_test_index] as usize;
                        if hierarchy_test_index == 0 {
                            break;
                        }
                        if hierarchy_test_index == a_child_node_index {
                            continue 'b_child;
                        }
                    }
                    let mut hierarchy_test_index = a_child_node_index;
                    loop {
                        hierarchy_test_index = parents[hierarchy_test_index] as usize;
                        if hierarchy_test_index == 0 {
                            break;
                        }
                        if hierarchy_test_index == b_child_node_index {
                            continue 'b_child;
                        }
                    }
                    let b_child_aabb = cwbvh.node_aabb(b_child_node_index);

                    let b_eval_aabb = b_parent_minus_this_ch.union(&a_child_aabb);
                    let a_eval_aabb = a_parent_minus_this_ch.union(&b_child_aabb);

                    let new_cost = b_eval_aabb.half_area() + a_eval_aabb.half_area();
                    let old_cost = a_parent_cost + b_parent_cost;
                    let new_diff = old_cost - new_cost;
                    if new_diff > best_cost_diff {
                        best_swap_parent = b_parent_idx;
                        best_swap_child_inside = b_node_ch;
                        best_swap_child = b_child_node_index;
                        best_cost_diff = new_diff;
                    }
                }
            }
            if best_cost_diff > 0.0 {
                assert!(!cwbvh.nodes[best_swap_parent].is_leaf(best_swap_child_inside));
                assert!(!cwbvh.nodes[a_parent_idx].is_leaf(a_node_ch));
                assert!(a_child_node_index != best_swap_child);
                assert!(parents[a_child_node_index] as usize != best_swap_child);
                assert!(a_child_node_index != parents[best_swap_child] as usize);
                let node_a = cwbvh.nodes[a_child_node_index];
                let node_b = cwbvh.nodes[best_swap_child];
                cwbvh.nodes[best_swap_child] = node_a;
                cwbvh.nodes[a_child_node_index] = node_b;
                if let Some(exact_node_aabbs) = &mut cwbvh.exact_node_aabbs {
                    assert!(exact_node_aabbs[a_child_node_index].contains_point(node_a.p.into()));
                    assert!(exact_node_aabbs[best_swap_child].contains_point(node_b.p.into()));
                    let node_aabb_a = exact_node_aabbs[a_child_node_index];
                    let node_aabb_b = exact_node_aabbs[best_swap_child];
                    exact_node_aabbs[a_child_node_index] = node_aabb_b;
                    exact_node_aabbs[best_swap_child] = node_aabb_a;
                    assert!(exact_node_aabbs[a_child_node_index].contains_point(node_b.p.into()));
                    assert!(exact_node_aabbs[best_swap_child].contains_point(node_a.p.into()));
                }
                //assert!(!cwbvh.nodes[a_parent_idx].is_leaf(a_node_ch));
                //assert!(!cwbvh.nodes[best_swap_parent].is_leaf(best_swap_child_inside));
                // TODO don't compute_parents, instead go to the children of each nodes in the parents list and update
                //parents = cwbvh.compute_parents();

                for ch in 0..8 {
                    if !node_a.is_child_empty(ch) {
                        if !node_a.is_leaf(ch) {
                            parents[node_a.child_node_index(ch) as usize] = best_swap_child as u32;
                        }
                    }
                    if !node_b.is_child_empty(ch) {
                        if !node_b.is_leaf(ch) {
                            parents[node_b.child_node_index(ch) as usize] =
                                a_child_node_index as u32;
                        }
                    }
                }

                cwbvh.refit(&parents, false, &primitives);
                //cwbvh.refit_from(a_child_node_index, &parents, false, true, &primitives);
                //cwbvh.refit_from(best_swap_child, &parents, false, true, &primitives);

                //for (child, _parent) in parents.iter().enumerate().take(1) {
                //    // This will use the exact aabb if they are included
                //    cwbvh.refit_from(child, &parents, false, true, &primitives);
                //}

                let node_a = cwbvh.nodes[best_swap_child];
                let node_b = cwbvh.nodes[a_child_node_index];
                let parent_a = cwbvh.nodes[parents[best_swap_child] as usize];
                let parent_b = cwbvh.nodes[parents[a_child_node_index] as usize];
                dbg!(a_child_node_index, best_swap_child);

                assert!(parent_a.aabb().contains_point(node_a.p.into()));
                assert!(parent_b.aabb().contains_point(node_b.p.into()));

                touched[a_child_node_index] = true;
                touched[best_swap_child] = true;
                dbg!(a_parent_idx);
                dbg!("SWAP!");
                //return;
            }
        }
    }
    dbg!("END cwbvh_reinsertion");
}

pub fn get_total_aabb_minus_one<T: Boundable>(
    cwbvh: &CwBvh,
    parent_index: usize,
    skip: usize,
    direct_layout: bool,
    primitives: &[T],
) -> Aabb {
    let node = &cwbvh.nodes[parent_index];
    let mut aabb = Aabb::empty();
    let mut full_aabb = Aabb::empty();
    for ch in 0..8 {
        //if ch == skip {
        //    continue;
        //}
        if node.is_child_empty(ch) {
            // TODO allow swapping with empty slots
            continue;
        }
        if node.is_leaf(ch) {
            let (child_prim_start, count) = node.child_primitives(ch);
            for i in 0..count {
                let mut prim_index = (child_prim_start + i) as usize;
                if !direct_layout {
                    prim_index = cwbvh.primitive_indices[prim_index] as usize;
                }
                if ch != skip {
                    aabb = aabb.union(&primitives[prim_index].aabb());
                }
                full_aabb = full_aabb.union(&primitives[prim_index].aabb());
            }
        } else {
            if ch != skip {
                aabb = aabb.union(&cwbvh.node_aabb(node.child_node_index(ch) as usize));
            }
            full_aabb = full_aabb.union(&cwbvh.node_aabb(node.child_node_index(ch) as usize));
        }
    }
    //assert!(full_aabb.half_area() == cwbvh.node_aabb(parent_index).half_area());
    aabb
}
