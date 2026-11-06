# Third-party notices

This example uses separately installed Genesis, Quadrants, PyTorch and their dependencies. Their names identify the software used by this test. They do not imply endorsement of this example. The repository does not redistribute an installed Python environment or a CUDA toolkit.

## Genesis

The native simulation interfaces and adapters target Genesis 1.3.2. Genesis is distributed under the Apache License, Version 2.0. The complete license text is provided in `licenses/Apache-2.0.txt`.

Source: [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world). License: [Genesis LICENSE](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/LICENSE).

## Franka Emika Panda model and geometry calibration

The Panda robot description and meshes come from the model supplied with Genesis. The corresponding [Franka Emika Panda description in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) documents its derivation from the original [Franka ROS description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description). The model README supplied with Genesis identifies the Apache License, Version 2.0.

Model attribution and license: [Panda model README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/README.md), [Panda model LICENSE](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/LICENSE), and the [model README supplied by Genesis](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/genesis/assets/xml/franka_emika_panda/README.md).

The file `physics/assets/fixed_peer_geometry_calibration_v1.json` contains model-derived collision vertices and link poses captured at specified initial joint configurations. It is a new geometry-calibration representation, rather than an unmodified upstream mesh file. The calibration changes coordinate representation and records model metadata. It does not change the upstream license or transfer ownership of the underlying model. The complete Apache-2.0 license text accompanies this data in `licenses/Apache-2.0.txt`.

Fixed-peer MJCF files are generated locally from the installed Panda model when the test runs. The generated files bake the supplied starting joint configuration into fixed link transforms. Original collision geometry, material parameters and collision masks are retained and checked against the supplied calibration. These generated files are execution output and are not part of the published source payload.

## Other installed dependencies

Quadrants, PyTorch and other installed packages retain their own licenses and notices. Refer to the license files included with those packages. This repository does not replace their licenses or claim ownership of their source, binaries or assets.

The supplied example path is fixed test input for this repository. It is not a general motion-planning algorithm.
