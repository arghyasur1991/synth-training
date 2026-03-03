# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-04

### Added
- SAC agent with TorchSharp: twin Q-networks, automatic entropy tuning, TD3-style target policy smoothing
- ContinuousLearningSkill implementing ISynthSkill for persistent on-device training
- Prioritized Experience Replay buffer with importance sampling and beta annealing
- Training thread with platform-adaptive throttling (Metal/MPS, Android CPU, Windows CPU)
- Observation normalizer with running mean/std
- Continuing reward with phase-based shaping (standing, recovering, moving)
- State persistence with atomic saves (crash-safe)
- Motion reference tooling: MotionClipExtractor, MotionReferenceData, ReferenceAnimationPlayer, MotionExtractionTestBench
- TorchSharpLoader for automatic native library bootstrapping on all platforms
- IL2CPP bridge for TorchSharp on Quest/Android
- Build scripts for macOS (setup_torchsharp_macos.sh) and Android arm64 (setup_torchsharp_android.sh)
- CMake cross-compilation tooling for LibTorchSharp on Android
