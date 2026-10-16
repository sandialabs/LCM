#!/usr/bin/env python3
"""
Imprint crystal lattice orientations on a genesis mesh as element attributes,
for CrystalPlasticity with "Read Lattice Orientation From Mesh: true".

    python3 imprint_orientation_on_mesh.py -m mesh.g -r mesh_Rotations.txt [-o out.g] [--combine-blocks]

Rotations file: one rotation matrix R per element block, in the order of the
blocks in the mesh file, written row by row: R11 R12 R13 R21 R22 R23 R31 R32 R33.
Lines starting with '#' are ignored. This is the same file that
write_file_material.py reads, where the columns of R are the lattice basis
vectors in global components ("Basis Vector 1/2/3").

Attribute convention: each element gets nine attributes, R written row by row
(R11 R12 R13 R21 ... R33), where the columns of R are the lattice basis
vectors in global components. LCM's CrystalPlasticity reads them back the same
way, so a mesh orientation and the same "Basis Vector 1/2/3" input describe
the same lattice. This is the layout of the meshes made with the original
version of this script (2016).

With --combine-blocks, all blocks are merged into a single block (they must
share the element type), and each element keeps the orientation of the block
it came from. Element order, and hence the element map and side sets, is
unchanged because the blocks are concatenated in file order.

To use the mesh, set "Read Lattice Orientation From Mesh: true" in the
material (at the same level as "Crystal Elasticity", not inside it) and declare
the field in the discretization:

    Required Fields Info:
      Number Of Fields: 1
      Field 0:
        Field Name: Lattice_Orientation
        Field Type: Elem Scalar
        Field Origin: Mesh

See tests/LCM/CrystalPlasticity/SingleElement3d/ElasticShearRotated3dMesh.yaml.

Requires exodus3.py (or exodus.py) from SEACAS on PYTHONPATH.
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import exodus3 as exodus
except ImportError:
    import exodus

NUM_ATTRIBUTES = 9


def read_rotations(file_name):
    rotations = np.loadtxt(file_name, comments='#', ndmin=2)
    if rotations.shape[1] != NUM_ATTRIBUTES:
        sys.exit(f'{file_name}: expected 9 values per line (a 3x3 rotation matrix), found {rotations.shape[1]}')
    rotations = rotations.reshape((-1, 3, 3))
    for k, R in enumerate(rotations):
        error = max(np.abs(R.T @ R - np.eye(3)).max(), abs(np.linalg.det(R) - 1.0))
        if error > 1.0e-6:
            sys.exit(f'{file_name}: matrix {k + 1} is not a proper rotation (error {error:.1e})')
    return rotations


def imprint(mesh_name, rotations_name, output_name, combine_blocks):
    mesh = exodus.exodus(mesh_name, mode='r', array_type='numpy')
    block_ids = list(mesh.get_elem_blk_ids())
    rotations = read_rotations(rotations_name)
    if len(rotations) != len(block_ids):
        sys.exit(f'{rotations_name} has {len(rotations)} rotations but {mesh_name} has {len(block_ids)} element blocks')
    if any(mesh.num_attr(b) > 0 for b in block_ids):
        sys.exit(f'{mesh_name}: meshes with existing element attributes are not supported')

    blocks = []
    for block_id, R in zip(block_ids, rotations):
        connectivity, num_elems, num_nodes_per_elem = mesh.get_elem_connectivity(block_id)
        # R row by row; its columns are the lattice basis vectors
        attributes = np.tile(R.flatten(), num_elems)
        blocks.append(dict(id=block_id, type=mesh.elem_type(block_id), num_elems=num_elems,
                           num_nodes=num_nodes_per_elem, connectivity=np.asarray(connectivity).flatten(),
                           attributes=attributes))
    block_names = list(mesh.get_elem_blk_names())

    if combine_blocks and len(blocks) > 1:
        types = {(b['type'], b['num_nodes']) for b in blocks}
        if len(types) > 1:
            sys.exit(f'--combine-blocks needs a single element type; found {sorted(types)}')
        blocks = [dict(id=blocks[0]['id'], type=blocks[0]['type'], num_elems=sum(b['num_elems'] for b in blocks),
                       num_nodes=blocks[0]['num_nodes'],
                       connectivity=np.concatenate([b['connectivity'] for b in blocks]),
                       attributes=np.concatenate([b['attributes'] for b in blocks]))]
        block_names = [block_names[0]]

    if os.path.exists(output_name):
        os.remove(output_name)
    out = exodus.exodus(output_name, mode='w', array_type='numpy', title=mesh.title(),
                        numDims=mesh.num_dimensions(), numNodes=mesh.num_nodes(), numElems=mesh.num_elems(),
                        numBlocks=len(blocks), numNodeSets=mesh.num_node_sets(), numSideSets=mesh.num_side_sets())

    qa_records = list(mesh.get_qa_records())
    qa_records.append(('imprint_orientation_on_mesh.py', '2.0', time.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')))
    out.put_qa_records(qa_records)

    out.put_coord_names(mesh.get_coord_names())
    out.put_coords(*mesh.get_coords())
    out.put_node_id_map(mesh.get_node_id_map())
    out.put_elem_id_map(mesh.get_elem_id_map())

    out.put_concat_elem_blk([b['id'] for b in blocks], [b['type'] for b in blocks],
                            [b['num_elems'] for b in blocks], [b['num_nodes'] for b in blocks],
                            [NUM_ATTRIBUTES for b in blocks], 1)
    for b in blocks:
        out.put_elem_connectivity(b['id'], b['connectivity'])
        out.put_elem_attr(b['id'], b['attributes'])
    out.put_elem_blk_names(block_names)

    node_set_ids = list(mesh.get_node_set_ids())
    for ns in node_set_ids:
        num_nodes, num_df = mesh.get_node_set_params(ns)
        out.put_node_set_params(ns, num_nodes, num_df)
        out.put_node_set(ns, mesh.get_node_set_nodes(ns))
        if num_df > 0:
            out.put_node_set_dist_fact(ns, mesh.get_node_set_dist_facts(ns))
    if node_set_ids:
        out.put_node_set_names(mesh.get_node_set_names())

    side_set_ids = list(mesh.get_side_set_ids())
    for ss in side_set_ids:
        num_sides, num_df = mesh.get_side_set_params(ss)
        out.put_side_set_params(ss, num_sides, num_df)
        elems, sides = mesh.get_side_set(ss)
        out.put_side_set(ss, elems, sides)
        if num_df > 0:
            out.put_side_set_dist_fact(ss, mesh.get_side_set_dist_fact(ss))
    if side_set_ids:
        out.put_side_set_names(mesh.get_side_set_names())

    out.close()
    mesh.close()
    print(f'Wrote {output_name}: {len(blocks)} block(s), {NUM_ATTRIBUTES} orientation attributes per element')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--meshfile', required=True, help='input genesis mesh')
    parser.add_argument('-r', '--rotfile', help='rotations file (default: <mesh base>_Rotations.txt)')
    parser.add_argument('-o', '--output', help='output mesh (default: <mesh base>_orientation.g)')
    parser.add_argument('--combine-blocks', action='store_true', help='merge all element blocks into one')
    args = parser.parse_args()

    base = os.path.splitext(args.meshfile)[0]
    rotfile = args.rotfile or base + '_Rotations.txt'
    output = args.output or base + '_orientation.g'
    imprint(args.meshfile, rotfile, output, args.combine_blocks)


if __name__ == '__main__':
    main()
