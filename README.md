# Synth Training

On-device reinforcement learning for Synth humanoids using [TorchSharp](https://github.com/dotnet/TorchSharp). Train directly in the Unity editor or on Meta Quest — no external Python server needed.

## Features

- **Dual Algorithm Support** — SAC (off-policy) and PPO (on-policy) via a shared `BaseTrainingSkill` abstraction. Choose the right algorithm for each task.
- **DeepMimic Imitation Learning** — `ImitationLearningSkill` tracks reference AnimationClips using pose, velocity, and key-body rewards with multi-clip support and hard negative mining.
- **Continuous Learning** — `ContinuousLearningSkill` for persistent, always-on training with phase-based reward shaping and contact-based micro-rewards.
- **Inference Mode** — Run trained policies without the training loop. Deterministic or stochastic action modes, with automatic model loading from saved checkpoints.
- **Platform-Adaptive** — macOS (Metal/MPS GPU), Android/Quest (CPU), Windows (CPU). Training thread auto-throttles based on platform capabilities.
- **Model Deployment Pipeline** — Automatic packaging of trained models into builds via `IPreprocessBuild`/`IPostprocessBuild` hooks. First-launch extraction on device via `ModelBootstrap`.
- **Double-Buffered CPU Inference** — PPO uses lock-free CPU inference clones, allowing the main thread to run inference while the training thread updates GPU weights concurrently.
- **Progressive Action Curriculum** — Unlock joints in stages as the agent improves, with automatic target entropy adjustment.
- **Live Training Dashboard** — Editor window (`Synth/Training Dashboard`) with real-time graphs for reward components, losses, alpha, SPS, and skill-specific diagnostics.
- **Motion Reference Tooling** — Extract reference motion from AnimationClips, play back on non-MuJoCo characters, and visually validate motion extraction pipelines.
- **Atomic State Persistence** — Crash-safe save/load with temporary file and atomic rename. Survives interrupted writes.
- **IL2CPP Compatible** — Custom bridge for TorchSharp on IL2CPP (Quest/Android). Static forward-slot pool avoids marshalling issues.

## Ecosystem

synth-training is part of a three-package architecture for creating, training, and interacting with physics-simulated humanoids:

| Package | Role | |
|---------|------|-|
| [**synth-core**](https://github.com/arghyasur1991/synth-core) | Humanoid creation, MuJoCo physics, skill architecture | Required |
| **synth-training** *(this repo)* | On-device RL training (SAC + PPO) and inference via TorchSharp | — |
| [**synth-vr**](https://github.com/arghyasur1991/synth-vr) | Mixed reality interaction on Meta Quest | Optional |

synth-core provides the physics body, motor system, and extensible skill/sense interfaces that synth-training builds on. This package implements `ISynthSkill` to add learning directly in Unity. When combined with **synth-vr**, training runs live on Meta Quest while you physically interact with the Synth in your room.

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

### Imitation Learning (PPO)

1. Set up a Synth using synth-core (see its README).
2. Add `ImitationLearningSkill` to your Synth prefab.
3. Assign one or more reference AnimationClips.
4. Press Play — PPO training tracks the reference motion using DeepMimic rewards.

### Continuous Learning (SAC)

1. Add `ContinuousLearningSkill` to your Synth prefab.
2. Configure SAC hyperparameters in the inspector.
3. Press Play — training begins automatically with contact-based rewards.

### Inference Only

1. Check **Inference Only** on any training skill component.
2. Optionally uncheck **Deterministic Inference** for stochastic (noisy) actions.
3. Press Play — the policy runs from saved weights without training.

### Model Deployment (Quest Builds)

Trained models are automatically packaged into builds:

1. Create `Assets/Resources/SynthBuildSettings.asset` via **Assets > Create > Synth > Build Settings** (auto-created on first build if missing).
2. Build for Android/Quest — models are copied from `persistentDataPath` to `StreamingAssets` pre-build.
3. On first launch, `ModelBootstrap` extracts models to `persistentDataPath` on the device.
4. `StreamingAssets` copies are cleaned up post-build (configurable).

## Architecture

```
BaseTrainingSkill (MonoBehaviour, ISynthSkill)
├── observe → BuildFullObs → normalize ALL → policy → action
├── Inference mode: obs → deterministic/stochastic action (no training loop)
├── Training mode: obs → action → reward → store → background train thread
│
├── ImitationLearningSkill (PPO)
│   ├── DeepMimic reward (pose, velocity, root, key-body)
│   ├── Multi-clip library with hard negative mining
│   └── Reference motion advancement via AdvanceTime()
│
└── ContinuousLearningSkill (SAC)
    ├── Contact-based micro-rewards
    ├── Progressive action curriculum
    └── Prioritized experience replay
```

## Package Structure

```
synth-training/
├── Runtime/
│   ├── Skills/            BaseTrainingSkill, ImitationLearningSkill,
│   │                      ContinuousLearningSkill
│   ├── Agent/             PPOAgent, SACAgent, StructuredActorNetwork
│   ├── Training/          ISkillTrainer, BaseSkillTrainer,
│   │                      PPOSkillTrainer, SACSkillTrainer,
│   │                      RolloutBuffer, ReplayBuffer, TrainingThread
│   ├── Reward/            DeepMimicReward, ContinuingReward
│   ├── Curriculum/        ActionCurriculum
│   ├── Observation/       ObservationNormalizer
│   ├── Build/             SynthBuildSettings, ModelBootstrap
│   ├── Diagnostics/       TrainingMetrics
│   ├── Persistence/       StatePersister
│   ├── MotionReference/   MotionClipExtractor, MotionReferenceData,
│   │                      ReferenceAnimationPlayer, MotionExtractionTestBench
│   └── Utility/           LearningLogger, TorchSharpLoader
├── Editor/
│   ├── TrainingDashboard.cs
│   ├── SynthModelBuildProcessor.cs
│   └── ContinuousLearningSkillEditor.cs
├── scripts/
│   ├── setup_torchsharp_macos.sh
│   └── setup_torchsharp_android.sh
└── tools~/
    └── torchsharp_android/   CMakeLists.txt, android_stubs.cpp
```

## Supported Platforms

| Platform | Device | Status |
|----------|--------|--------|
| macOS Metal (MPS) | Mac editor | GPU training + CPU inference |
| Android CPU | Meta Quest 3 | Throttled training, inference mode |
| Windows CPU | Windows editor | Supported |

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
