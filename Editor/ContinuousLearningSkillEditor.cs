using UnityEditor;
using UnityEngine;

namespace Genesis.Sentience.Learning
{
    [CustomEditor(typeof(ContinuousLearningSkill))]
    public class ContinuousLearningSkillEditor : Editor
    {
        // Phase distribution tracking
        private int[] _phaseCounts = new int[4];
        private int _phaseTotal;
        private float _lastPhaseUpdateTime;

        // Reward EMA for display
        private float _rewardEMA;
        private bool _rewardEMAInitialized;

        // Root Z tracking
        private float _rootZ;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var skill = (ContinuousLearningSkill)target;

            if (skill.deleteSavesOnStart)
            {
                EditorGUILayout.HelpBox(
                    "All saved learning state will be DELETED on next Play. " +
                    "This includes networks, replay buffer, normalizer, and physics state. " +
                    "Uncheck to cancel.", MessageType.Warning);
            }

            if (!skill.IsReady)
            {
                if (!Application.isPlaying)
                {
                    EditorGUILayout.HelpBox("Press Play to start continuous learning.", MessageType.Info);
                }
                else
                {
                    EditorGUILayout.HelpBox("Skill not initialized. Ensure " +
                        "SynthBrain discovers this component.", MessageType.Info);
                }
                return;
            }

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Continuous Learning Status", EditorStyles.boldLabel);

            // ── Decisions & Buffer ──
            EditorGUILayout.LabelField("Total Decisions", skill.TotalDecisions.ToString("N0"));
            EditorGUILayout.LabelField("Replay Buffer", $"{skill.ReplayBufferCount:N0}");
            EditorGUILayout.LabelField("Train Steps", skill.TrainSteps.ToString("N0"));

            EditorGUILayout.Space(5);

            // ── Training Metrics ──
            EditorGUILayout.LabelField("Training Metrics", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Training SPS",
                skill.TrainingSPS > 0 ? $"{skill.TrainingSPS:F1}" : "waiting...");
            EditorGUILayout.LabelField("Alpha (entropy)", $"{skill.Alpha:F4}");
            EditorGUILayout.LabelField("Q Loss", $"{skill.LastQLoss:F4}");
            EditorGUILayout.LabelField("Actor Loss", $"{skill.LastActorLoss:F4}");

            EditorGUILayout.Space(5);

            // ── Reward ──
            EditorGUILayout.LabelField("Reward", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Raw Reward", $"{skill.RawReward:F4}");
            EditorGUILayout.LabelField("Centered Reward", $"{skill.CenteredReward:F4}");
            EditorGUILayout.LabelField("Reward Bar (avg)", $"{skill.RewardBar:F4}");
            EditorGUILayout.LabelField("Nearest Frame Dist", $"{skill.NearestFrameDistance:F4}");

            EditorGUILayout.Space(5);

            // ── Phase ──
            EditorGUILayout.LabelField("Agent Phase", EditorStyles.boldLabel);
            var phase = skill.CurrentPhase;
            var phaseColor = phase switch
            {
                AgentPhase.Fallen => Color.red,
                AgentPhase.Recovering => Color.yellow,
                AgentPhase.Standing => Color.green,
                AgentPhase.Moving => Color.cyan,
                _ => Color.white
            };
            var prevColor = GUI.contentColor;
            GUI.contentColor = phaseColor;
            EditorGUILayout.LabelField("Current Phase", phase.ToString());
            GUI.contentColor = prevColor;

            // Phase distribution
            if (Application.isPlaying)
            {
                _phaseCounts[(int)phase]++;
                _phaseTotal++;

                if (_phaseTotal > 0)
                {
                    float pFallen = _phaseCounts[0] * 100f / _phaseTotal;
                    float pRecovering = _phaseCounts[1] * 100f / _phaseTotal;
                    float pStanding = _phaseCounts[2] * 100f / _phaseTotal;
                    float pMoving = _phaseCounts[3] * 100f / _phaseTotal;

                    EditorGUILayout.LabelField("Phase Distribution",
                        $"F:{pFallen:F0}% R:{pRecovering:F0}% S:{pStanding:F0}% M:{pMoving:F0}%");
                }
            }

            // ── Assisted Poses ──
            if (skill.enableAssistedPoses)
            {
                EditorGUILayout.Space(5);
                EditorGUILayout.LabelField("Assisted Poses", EditorStyles.boldLabel);
                EditorGUILayout.LabelField("Teleport Count", skill.AssistCount.ToString("N0"));
                EditorGUILayout.LabelField("Fallen Timer",
                    $"{skill.FallenTimer:F1}s / {skill.assistIntervalSeconds:F0}s");
                if (skill.AssistHoldRemaining > 0)
                    EditorGUILayout.LabelField("Holding Pose",
                        $"{skill.AssistHoldRemaining} steps remaining");
            }

            EditorGUILayout.Space(10);

            // ── Actions ──
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Save State"))
                skill.SaveState();
            if (GUILayout.Button("Reset Phase Stats"))
            {
                _phaseCounts = new int[4];
                _phaseTotal = 0;
            }
            EditorGUILayout.EndHorizontal();

            if (Application.isPlaying)
                Repaint();
        }
    }
}
