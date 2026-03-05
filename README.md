# Synth Training

On-device reinforcement learning for Synth humanoids using [TorchSharp](https://github.com/dotnet/TorchSharp). Train directly in the Unity editor or on Meta Quest — no external Python server needed.

## Features

- **On-Device SAC Training** — Soft Actor-Critic with twin Q-networks, automatic entropy tuning, and TD3-style target policy smoothing.
- **Platform-Adaptive** — macOS (Metal/MPS), Android/Quest (CPU), Windows (CPU). Training thread auto-throttles based on platform capabilities.
- **Prioritized Experience Replay** — PER with importance sampling and beta annealing for sample-efficient learning.
- **Continuous Learning** — `ContinuousLearningSkill` implements `ISynthSkill` for persistent, always-on training with phase-based reward shaping.
- **Progressive Action Curriculum** — Unlock joints in stages as the agent improves, with automatic target entropy adjustment.
- **Live Training Dashboard** — Editor window (`Synth/Training Dashboard`) with real-time graphs for reward components, losses, alpha, phase timeline, and performance metrics.
- **Motion Reference Tooling** — Extract reference motion from AnimationClips, play back on non-MuJoCo characters, and visually validate motion extraction pipelines.
- **Atomic State Persistence** — Crash-safe save/load with temporary file and atomic rename. Survives interrupted writes.
- **IL2CPP Compatible** — Custom bridge for TorchSharp on IL2CPP (Quest/Android). Static forward-slot pool avoids marshalling issues.

## Ecosystem

synth-training is part of a three-package architecture for creating, training, and interacting with physics-simulated humanoids:

| Package | Role | |
|---------|------|-|
| [**synth-core**](https://github.com/arghyasur1991/synth-core) | Humanoid creation, MuJoCo physics, skill architecture | Required |
| **synth-training** *(this repo)* | On-device reinforcement learning via TorchSharp SAC | — |
| [**synth-vr**](https://github.com/arghyasur1991/synth-vr) | Mixed reality interaction on Meta Quest | Optional |

synth-core provides the physics body, motor system, and extensible skill/sense interfaces that synth-training builds on. This package implements `ISynthSkill` to add continuous learning directly in Unity — no external Python server needed. When combined with **synth-vr**, training runs live on Meta Quest while you physically interact with the Synth in your room.

## Requirements

- Unity 6000.x or later
- [synth-core](https://github.com/arghyasur1991/synth-core) package
- MuJoCo Unity plugin (`org.mujoco`) — via [arghyasur1991/mujoco](https://github.com/arghyasur1991/mujoco) fork (`synth-patches` branch)
- [TorchSharp](https://github.com/arghyasur1991/TorchSharp) fork (`unity-il2cpp-support` branch) — includes IL2CPP bridge for Quest/Android
- Platform-specific native LibTorch libraries (see build instructions below)

### Build Prerequisites (for native libraries)

| Requirement | Purpose |
|-------------|---------|
| .NET SDK 8+ | Build TorchSharp managed DLL |
| CMake 3.18+ | Cross-compile LibTorchSharp for Android |
| Android NDK r26+ | Android arm64 cross-compilation |
| PyTorch source (v2.7.1) | Build LibTorch for Android (via submodule or clone) |

## Installation

Add to `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.genesis.synth.training": "https://github.com/arghyasur1991/synth-training.git",
    "com.genesis.synth": "https://github.com/arghyasur1991/synth-core.git",
    "org.mujoco": "https://github.com/arghyasur1991/mujoco.git?path=unity#synth-patches"
  }
}
```

### Native Libraries

TorchSharp requires platform-specific native libraries. Build and deploy using the included scripts:

```bash
# macOS (builds TorchSharp from source, deploys to Unity project)
./scripts/setup_torchsharp_macos.sh /path/to/YourUnityProject

# Android arm64 (cross-compiles LibTorch + LibTorchSharp)
./scripts/setup_torchsharp_android.sh /path/to/YourUnityProject
```

| Platform | Libraries | Deployment Location |
|----------|-----------|---------------------|
| macOS arm64 | `libtorch.dylib`, `libtorch_cpu.dylib`, `libc10.dylib`, `libLibTorchSharp.dylib` | `Assets/Plugins/arm64/` |
| Android arm64 | `libLibTorchSharp.so` | `Assets/Plugins/Android/arm64-v8a/` |

The managed `TorchSharp.dll` is deployed to `Assets/Packages/TorchSharp/`.

## Quick Start

1. Set up a Synth using synth-core (see its README).
2. Add the `ContinuousLearningSkill` component to your Synth prefab.
3. Configure SAC hyperparameters in the inspector.
4. Press Play — training begins automatically.

## Package Structure

```
synth-training/
├── Runtime/
│   ├── Skills/            ContinuousLearningSkill
│   ├── Agent/             SACAgent, SACConfig, SoftQNetwork, StructuredActorNetwork
│   ├── Training/          TrainingThread, ReplayBuffer
│   ├── Reward/            ContinuingReward (multi-phase, contact, proximity)
│   ├── Curriculum/        ActionCurriculum (progressive joint unlocking)
│   ├── Diagnostics/       TrainingMetrics, MetricRingBuffer
│   ├── Observation/       ObservationNormalizer
│   ├── Persistence/       StatePersister
│   ├── MotionReference/   MotionClipExtractor, MotionReferenceData,
│   │                      ReferenceAnimationPlayer, MotionExtractionTestBench
│   └── Utility/           LearningLogger, TorchSharpLoader
├── Editor/
│   ├── ContinuousLearningSkillEditor.cs
│   └── TrainingDashboard.cs
├── scripts/
│   ├── setup_torchsharp_macos.sh
│   └── setup_torchsharp_android.sh
└── tools~/
    └── torchsharp_android/   CMakeLists.txt, android_stubs.cpp
```

## Supported Platforms

| Platform | Device | Status |
|----------|--------|--------|
| macOS Metal (MPS) | Mac editor | Full speed |
| Android CPU | Meta Quest 3 | Throttled for thermal management |
| Windows CPU | Windows editor | Supported |

## Roadmap

- **Better continuous learning** — Improved reward shaping, locomotion emergence, homeostatic regulation, and phase progression beyond standing recovery
- **Imitation learning** — Learn from motion capture clips using adversarial or tracking-based reward
- **PPO support** — Proximal Policy Optimization as an alternative to SAC for on-policy training
- **Multi-agent training** — Train multiple Synths in parallel within a single Unity scene
- **Reward designer** — Visual editor for composing reward functions from observation primitives

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
