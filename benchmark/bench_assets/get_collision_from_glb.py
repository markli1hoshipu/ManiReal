import trimesh
import coacd
import numpy as np
import os

def process_collision_meshes(input_paths, threshold=0.02, shrink_factor=1):
    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Skipping: {input_path} (File not found)")
            continue

        print(f"Processing: {input_path}...")
        
        # 1. Load the original GLB
        mesh = trimesh.load(input_path, force='mesh') 

        # 2. Wrap into CoACD Mesh
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

        # 3. Run CoACD
        # Note: using positional arguments for max_convex_hull if keyword fails
        parts = coacd.run_coacd(
            coacd_mesh, 
            threshold=threshold
        )

        # 4. Apply Shrink
        hulls = []
        for p in parts:
            hull = trimesh.Trimesh(vertices=p[0], faces=p[1])
            center = hull.centroid
            
            # Scale vertices toward the center to create the "thinner" mesh
            new_vertices = center + (hull.vertices - center) * shrink_factor
            hull.vertices = new_vertices
            hulls.append(hull)

        # 5. Generate output path and export
        # Example: /path/link_0.glb -> /path/link_0_collision.glb
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_collision{ext}"
        
        collision_scene = trimesh.Scene(hulls)
        collision_scene.export(output_path)

        print(f"Done! Exported {len(hulls)} hulls to {output_path}")

# --- Configuration ---
paths_to_process = [
    # '/home/aaron/sajjad/robotwin_bench/customized_robotwin/assets/objects_bench/124_fridge_hivvdf/blender_public/links/link_0.glb',
    # '/home/aaron/sajjad/robotwin_bench/customized_robotwin/assets/objects_bench/124_fridge_hivvdf/blender_public/links/base_link.glb',
    # Add more paths here as needed
    '/home/aaron/sajjad/robotwin_bench/customized_robotwin/assets/objects_bench/125_cabinet_tynnnw/blender_public/links/base_link.glb',
    '/home/aaron/sajjad/robotwin_bench/customized_robotwin/assets/objects_bench/125_cabinet_tynnnw/blender_public/links/leftdoor.glb',
    '/home/aaron/sajjad/robotwin_bench/customized_robotwin/assets/objects_bench/125_cabinet_tynnnw/blender_public/links/rightdoor.glb',

]

if __name__ == "__main__":
    process_collision_meshes(paths_to_process)