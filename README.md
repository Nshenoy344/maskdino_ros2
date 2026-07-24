# maskdino_ros2





A ROS 2 (Humble) wrapper around MaskDINO for real-time instance segmentation and object detection on a live camera stream. The node subscribes to an image topic, runs MaskDINO inference (via Detectron2) on each frame, and publishes 2D detections/masks — optionally lifting each detection to a 3D pose using an aligned depth/point-cloud topic and TF.



## Quick start (Docker)



The included Docker setup builds a container with ROS 2 Humble, PyTorch, Detectron2, and MaskDINO already installed, and mounts your model/source files in.



```bash

cd maskdino_ros2/docker

docker compose build

docker compose up -d

docker exec -it maskdino_ros2 bash

```



`docker-compose.yml` mounts the following host directories into the container, so create them alongside the repo before starting:
 
- `../maskdino_ros_pkg/` → `/root/ros2_ws/src/maskdino_ros_pkg` — live-editable ROS 2 package source
- `../example_imgs` → `/imgs` — sample images
- `../outputs/` → `/outputs/` — training/inference outputs
- `../source/` → `/source/` — model weights, config, and labels (see `params.yaml`)



The container also requires an X server for visualization windows and NVIDIA runtime support (`runtime: nvidia`), and shares the host network (`network_mode: host`) so ROS 2 discovery works transparently.



On first start, `nvidia_entrypoint.sh` builds MaskDINO's MSDeformAttn CUDA pixel-decoder extension (this step can't be done at image-build time and must run once the GPU runtime is available).



Inside the container, build and source the workspace, then launch the node:



```bash

cd /root/ros2_ws

colcon build

source install/setup.bash

ros2 launch maskdino_ros_pkg mask_dino_ros_launch.py

```



## Configuration



Parameters live in `maskdino_ros_pkg/params/params.yaml` and are loaded by the launch file.





## Running



```bash

ros2 launch maskdino_ros_pkg mask_dino_ros_launch.py

```

