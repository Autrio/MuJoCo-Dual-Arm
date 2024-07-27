#!/bin/bash

if ! command -v yq &> /dev/null
then
    echo "yq could not be found. Please install it to proceed."
    exit 1
fi

name=$(yq -r '.parameters.name' config.yaml)
msc=$(yq -r '.parameters.ACD.Mujoco_scale' config.yaml)

tol=$(yq -r '.parameters.pnp.tolerance' config.yaml)
viz=$(yq -r '.parameters.pnp.visual' config.yaml)
Gidx=$(yq -r '.parameters.pnp.grasp_idx' config.yaml)

osc=$(python3 -m grasps.generate.grasp -o $name)
if [ $? -ne 0 ]; then
  echo "Grasp generation runtime call"
  exit 1
fi

if ! [[ "$osc" =~ ^[+-]?[0-9]+\.?[0-9]*$ ]]; then
  echo "Invalid Scale Datatype"
  exit 1
fi

python3 -m models.utils.ACD -o $name -os $osc -ms $msc
if [ $? -ne 0 ]; then
  echo "Approximate Convex Decomposition runtime call failed"
  exit 1
fi

python3 -m examples.pnp -d dual -t $tol -m $viz -o $name -ms $msc -g $Gidx
if [ $? -ne 0 ]; then
  echo "Controller runtime call failed"
  exit 1
fi