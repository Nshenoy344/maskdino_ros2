\# maskdino\_ros2





A ROS 2 (Humble) wrapper around MaskDINO for real-time instance segmentation and object detection on a live camera stream. The node subscribes to an image topic, runs MaskDINO inference (via Detectron2) on each frame, and publishes 2D detections/masks — optionally lifting each detection to a 3D pose using an aligned depth/point-cloud topic and TF.



\## Quick start (Docker)



The included Docker setup builds a container with ROS 2 Humble, PyTorch, Detectron2, and MaskDINO already installed, and mounts your model/source files in.



```bash

cd maskdino\_ros2/docker

docker compose build

docker compose up -d

docker exec -it maskdino\_ros2 bash

```



`docker-compose.yml` mounts the following host directories into the container, so create them alongside the repo before starting:



| Host path         | Container path                        | Purpose                                   |

|--------------------|---------------------------------------|--------------------------------------------|

| `../maskdino\_ros\_pkg/` | `/root/ros2\_ws/src/maskdino\_ros\_pkg` | Live-editable ROS 2 package source          |

| `../example\_imgs`  | `/imgs`                               | Sample images                               |

| `../outputs/`       | `/outputs/`                           | Training/inference outputs                  |

| `../source/`        | `/source/`                            | Model weights, config, and labels (see `params.yaml`) |



The container also requires an X server for visualization windows and NVIDIA runtime support (`runtime: nvidia`), and shares the host network (`network\_mode: host`) so ROS 2 discovery works transparently.



On first start, `nvidia\_entrypoint.sh` builds MaskDINO's MSDeformAttn CUDA pixel-decoder extension (this step can't be done at image-build time and must run once the GPU runtime is available).



Inside the container, build and source the workspace, then launch the node:



```bash

cd /root/ros2\_ws

colcon build

source install/setup.bash

ros2 launch maskdino\_ros\_pkg mask\_dino\_ros\_launch.py

```



\## Configuration



Parameters live in `maskdino\_ros\_pkg/params/params.yaml` and are loaded by the launch file.





\## Running



```bash

ros2 launch maskdino\_ros\_pkg mask\_dino\_ros\_launch.py

```

