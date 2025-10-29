//
// Created by 42yea on 2022/6/22.
//

#include "BVH.cuh"
#include <glm/glm.hpp>
#include <stack>
#include <algorithm>


struct Bucket {
    int axis = 0;
    float x = 0.0f;
    float sah = 0.0f;
    bool leaf = false;
    BBox left = {}, right = {};
};


BVH::BVH(const Scene &scene, int n_buckets) {
    BBox bbox;
    for (const auto &obj : scene.get_objects()) {
        const glm::mat4 &trans = obj.transform;
        const std::vector<Vertex> &verts = obj.model->get_vertices();

        for (int i = 0; i < verts.size() / 3; i++) {
            glm::vec3 a = glm::vec3(trans * glm::vec4(verts[i * 3 + 0].position, 1.0f));
            glm::vec3 b = glm::vec3(trans * glm::vec4(verts[i * 3 + 1].position, 1.0f));
            glm::vec3 c = glm::vec3(trans * glm::vec4(verts[i * 3 + 2].position, 1.0f));
            Triangle tri(a, b, c);
            primitives.push_back(tri);
            bbox = bbox + tri.bbox;
        }
    }
    // the whole scene bbox
    Node &root = nodes.emplace_back(0, (int) primitives.size(), -1, -1, bbox);

    if (n_buckets == 0) {
        return;
    }

    std::stack<int> stack;
    stack.push(0);

    while (!stack.empty()) {
        int node_idx = stack.top();
        stack.pop();
        Node node = nodes[node_idx];
        bbox = node.bbox;
        glm::vec3 span = bbox.span();
        glm::vec3 stride = span / (float) (n_buckets + 2);

        // generate buckets: 3 axis, each n buckets. Or the other way around. Whatever
        std::vector<Bucket> buckets;
        for (int i = 0; i < n_buckets; i++) {
            Bucket along_x = { 0, node.bbox.min.x + stride.x * (i + 1), 0.0f, false, {}, {} };
            Bucket along_y = { 1, node.bbox.min.y + stride.y * (i + 2), 0.0f, false, {}, {} };
            Bucket along_z = { 2, node.bbox.min.z + stride.z * (i + 3), 0.0f, false, {}, {} };
            buckets.insert(buckets.end(), { along_x, along_y, along_z });
        }

        // process buckets; calculate SAH
        Bucket &best_bucket = buckets[0];
        for (Bucket &bucket : buckets) {
            int n_left = 0, n_right = 0;

            for (int i = node.start; i < node.start + node.size; i++) {
                const Triangle &tri = primitives[i];
                glm::vec3 center = tri.center;
                bool goto_left = false;
                switch (bucket.axis) {
                    case 0:
                        if (center.x < bucket.x) {
                            goto_left = true;
                        }
                        break;

                    case 1:
                        if (center.y < bucket.x) {
                            goto_left = true;
                        }
                        break;

                    case 2:
                        if (center.z < bucket.x) {
                            goto_left = true;
                        }
                        break;

                    default:
                        throw std::exception("Unsupported dimension");
                }
                if (goto_left) {
                    n_left++;
                    bucket.left = bucket.left + tri.bbox;
                } else {
                    n_right++;
                    bucket.right = bucket.right + tri.bbox;
                }
            }

            // calculate SAH
            bucket.sah = 1.0f + bucket.left.surface_area() / bbox.surface_area() * n_left +
                    bucket.right.surface_area() / bbox.surface_area() * n_right;

            if (n_left == 0 || n_right == 0) {
                bucket.sah = std::numeric_limits<float>::max();
                bucket.leaf = true;
            }

            if (bucket.sah < best_bucket.sah) {
                best_bucket = bucket;
            }
        }

        if (best_bucket.leaf) {
            // no point in making new nodes
            continue;
        }

        // create nodes based on best bucket
        auto split = std::partition(primitives.begin() + node.start,
                                    primitives.begin() + (node.start + node.size), [&](const auto &prim) {
            switch (best_bucket.axis) {
                case 0:
                    return prim.center.x < best_bucket.x;

                case 1:
                    return prim.center.y < best_bucket.x;

                case 2:
                    return prim.center.z < best_bucket.x;

                default:
                    return true;
            }
        });

        int left_size = std::distance(primitives.begin() + node.start, split);
        int right_size = node.size - left_size;

        Node left_node(node.start, left_size, -1, -1, best_bucket.left);
        Node right_node(node.start + left_size, right_size, -1, -1, best_bucket.right);

        nodes.push_back(left_node);
        nodes.push_back(right_node);

        nodes[node_idx].l = nodes.size() - 2;
        nodes[node_idx].r = nodes.size() - 1;

        stack.push(nodes.size() - 2);
        stack.push(nodes.size() - 1);

    }
}

const std::vector<BVH::Node> &BVH::get_nodes() const {
    return nodes;
}

BVH::Node::Node() : start(0), size(0), l(0), r(0) {

}

BVH::Node::Node(int start, int size, int l, int r, BBox bbox) : start(start), size(size), l(l), r(r), bbox(bbox) {

}
