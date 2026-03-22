using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using Genesis.Sentience.Learning;

namespace Genesis.Sentience.Learning.EditorTools
{
    public class TrainingDashboard : EditorWindow
    {
        [MenuItem("Synth/Training Dashboard")]
        static void Open()
        {
            var w = GetWindow<TrainingDashboard>("Training Dashboard");
            w.minSize = new Vector2(600, 400);
        }

        private BaseTrainingSkill _skill;
        private BaseTrainingSkill[] _allSkills;
        private int _selectedSkillIdx;
        private Vector2 _scroll;

        private static readonly string[] WindowLabels = { "1 min", "5 min", "15 min", "30 min" };
        private static readonly int[] WindowSamples = { 600, 3000, 9000, 18000 };
        private int _windowIdx = 1;

        private bool _foldReward = true;
        private bool _foldMain = true;
        private bool _foldRecovery = true;
        private bool _foldContact = true;
        private bool _foldTraining = true;
        private bool _foldAlpha = true;
        private bool _foldState = true;
        private bool _foldPerf = true;
        private bool _foldWorldModel = true;
        private bool _foldDynamic = true;
        private bool _foldV2Reward = true;
        private bool _foldV2Progress = true;
        private bool _foldDragForce = true;

        static readonly Color C_RAW = new Color(0.30f, 0.90f, 0.35f);
        static readonly Color C_CENTERED = new Color(0.40f, 0.70f, 1.00f);
        static readonly Color C_BAR = new Color(0.60f, 0.60f, 0.60f, 0.6f);

        static readonly Color C_HEIGHT = new Color(0.30f, 0.69f, 0.31f);
        static readonly Color C_ORIENT = new Color(0.13f, 0.59f, 0.95f);
        static readonly Color C_IMIT = new Color(0.00f, 0.74f, 0.83f);
        static readonly Color C_COMFORT = new Color(0.91f, 0.12f, 0.39f);
        static readonly Color C_ENERGY = new Color(1.00f, 0.76f, 0.03f);
        static readonly Color C_ALIVE = new Color(0.62f, 0.62f, 0.62f);

        static readonly Color C_RECOVERY = new Color(1.00f, 0.60f, 0.00f);
        static readonly Color C_VEL_UP = new Color(0.61f, 0.15f, 0.69f);
        static readonly Color C_PHASE_B = Color.white;

        static readonly Color C_FOOT = new Color(0.55f, 0.76f, 0.29f);
        static readonly Color C_HAND = new Color(0.01f, 0.66f, 0.96f);
        static readonly Color C_ACTIVE = new Color(1.00f, 0.34f, 0.13f);

        static readonly Color C_QLOSS = new Color(0.96f, 0.26f, 0.21f);
        static readonly Color C_ALOSS = new Color(1.00f, 0.60f, 0.00f);
        static readonly Color C_ALPHA_L = new Color(1.00f, 0.92f, 0.23f);
        static readonly Color C_ALPHA = new Color(0.00f, 0.59f, 0.53f);

        static readonly Color C_ROOTZ = new Color(1.00f, 0.92f, 0.23f);
        static readonly Color C_BLEND = new Color(0.13f, 0.59f, 0.95f);
        static readonly Color C_PHASE_L = new Color(0.80f, 0.80f, 0.80f);

        static readonly Color C_WMLOSS = new Color(0.00f, 0.74f, 0.83f);
        static readonly Color C_PROGRESS = new Color(0.40f, 0.90f, 0.40f);
        static readonly Color C_CONTACT_R = new Color(0.01f, 0.66f, 0.96f);
        static readonly Color C_DRAG = new Color(1.00f, 0.47f, 0.00f);

        static readonly Color C_SPS = new Color(0.30f, 0.90f, 0.35f);
        static readonly Color C_BUF = new Color(0.40f, 0.70f, 1.00f);

        static readonly Color BG_GRAPH = new Color(0.11f, 0.11f, 0.13f);
        static readonly Color BG_GRID = new Color(0.20f, 0.20f, 0.22f);

        static readonly Color[] PhaseColors =
        {
            new Color(0.96f, 0.26f, 0.21f), // Fallen
            new Color(1.00f, 0.76f, 0.03f), // Recovering
            new Color(0.30f, 0.69f, 0.31f), // Standing
            new Color(0.13f, 0.59f, 0.95f), // Moving
        };

        static readonly Color[] DynamicPalette =
        {
            new Color(0.30f, 0.90f, 0.35f),
            new Color(0.40f, 0.70f, 1.00f),
            new Color(1.00f, 0.60f, 0.00f),
            new Color(0.61f, 0.15f, 0.69f),
            new Color(0.00f, 0.74f, 0.83f),
            new Color(0.96f, 0.26f, 0.21f),
            new Color(1.00f, 0.92f, 0.23f),
            new Color(0.91f, 0.12f, 0.39f),
            new Color(0.55f, 0.76f, 0.29f),
            new Color(0.62f, 0.62f, 0.62f),
        };

        void OnEnable()
        {
            EditorApplication.playModeStateChanged += OnPlayMode;
        }

        void OnDisable()
        {
            EditorApplication.playModeStateChanged -= OnPlayMode;
        }

        void OnPlayMode(PlayModeStateChange _)
        {
            _skill = null;
            _allSkills = null;
        }

        void Update()
        {
            if (!EditorApplication.isPlaying) { _skill = null; return; }
            if (_skill == null || !_skill.IsReady) RefreshSkills();
            Repaint();
        }

        void RefreshSkills()
        {
            _allSkills = FindObjectsByType<BaseTrainingSkill>(
                FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            if (_allSkills.Length > 0)
            {
                _selectedSkillIdx = Math.Min(_selectedSkillIdx, _allSkills.Length - 1);
                _skill = _allSkills[_selectedSkillIdx];
            }
            else
            {
                _skill = null;
            }
        }

        void OnGUI()
        {
            DrawToolbar();

            if (!EditorApplication.isPlaying)
            {
                EditorGUILayout.HelpBox("Enter Play mode to see live training data.", MessageType.Info);
                return;
            }

            if (_skill == null || !_skill.IsReady || _skill.Metrics == null)
            {
                EditorGUILayout.HelpBox("Waiting for a training skill to initialize...", MessageType.Info);
                return;
            }

            var m = _skill.Metrics;
            int window = WindowSamples[_windowIdx];

            bool isContinuous = _skill is ContinuousLearningSkill;
            bool isV2 = _skill is ContinuousLearningSkillV2;

            DrawSummary(m, isContinuous, isV2);

            _scroll = EditorGUILayout.BeginScrollView(_scroll);

            if (isContinuous && m.Phase.Count > 1)
                DrawPhaseStrip(m, window);

            // Reward overview (universal)
            if (_foldReward = EditorGUILayout.Foldout(_foldReward, "Reward Overview", true, EditorStyles.foldoutHeader))
            {
                if (isContinuous)
                    DrawGraph(150, window,
                        (m.RawReward, C_RAW, "Raw"),
                        (m.CenteredReward, C_CENTERED, "Centered"),
                        (m.RewardBar, C_BAR, "Bar"));
                else
                    DrawGraph(150, window,
                        (m.RawReward, C_RAW, "Raw"));
            }

            // ContinuousLearning-specific sections
            if (isContinuous)
            {
                if (_foldMain = EditorGUILayout.Foldout(_foldMain, "Main Reward Components", true, EditorStyles.foldoutHeader))
                    DrawGraph(170, window,
                        (m.Height, C_HEIGHT, "Height"),
                        (m.Orientation, C_ORIENT, "Orient"),
                        (m.Imitation, C_IMIT, "Imitation"),
                        (m.Comfort, C_COMFORT, "Comfort"),
                        (m.Energy, C_ENERGY, "Energy"),
                        (m.Alive, C_ALIVE, "Alive"));

                if (_foldRecovery = EditorGUILayout.Foldout(_foldRecovery, "Recovery Rewards", true, EditorStyles.foldoutHeader))
                    DrawGraph(150, window,
                        (m.Recovery, C_RECOVERY, "Recovery"),
                        (m.VelocityUp, C_VEL_UP, "VelUp"),
                        (m.PhaseBonus, C_PHASE_B, "PhaseBonus"));

                if (_foldContact = EditorGUILayout.Foldout(_foldContact, "Contact Rewards", true, EditorStyles.foldoutHeader))
                    DrawGraph(150, window,
                        (m.FootSupport, C_FOOT, "Foot"),
                        (m.HandBrace, C_HAND, "Hand"),
                        (m.ActiveSupport, C_ACTIVE, "Active"));

                if (_foldTraining = EditorGUILayout.Foldout(_foldTraining, "Training Losses (SAC)", true, EditorStyles.foldoutHeader))
                    DrawGraph(150, window,
                        (m.QLoss, C_QLOSS, "QLoss"),
                        (m.ActorLoss, C_ALOSS, "ActorLoss"),
                        (m.AlphaLoss, C_ALPHA_L, "AlphaLoss"));

                if (_foldAlpha = EditorGUILayout.Foldout(_foldAlpha, "Alpha (Entropy Temperature)", true, EditorStyles.foldoutHeader))
                    DrawGraph(120, window,
                        (m.Alpha, C_ALPHA, "Alpha"));

                if (m.WorldModelLoss.Count > 0)
                {
                    if (_foldWorldModel = EditorGUILayout.Foldout(_foldWorldModel, "World Model (Dreaming)", true, EditorStyles.foldoutHeader))
                    {
                        DrawGraph(120, window,
                            (m.WorldModelLoss, C_WMLOSS, "WM Loss"));

                        var cls = (ContinuousLearningSkill)_skill;
                        var sacTrainer = cls.Trainer as SACSkillTrainer;
                        if (sacTrainer != null)
                        {
                            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
                            Lbl($"WM Loss: {sacTrainer.LastWorldModelLoss:F4}", 120);
                            Lbl($"Dreams: {sacTrainer.DreamPhaseCount}", 80);
                            GUILayout.FlexibleSpace();
                            EditorGUILayout.EndHorizontal();
                        }
                    }
                }

                if (!isV2)
                {
                    if (_foldState = EditorGUILayout.Foldout(_foldState, "Agent State", true, EditorStyles.foldoutHeader))
                        DrawGraph(150, window,
                            (m.RootZ, C_ROOTZ, "RootZ"),
                            (m.StandBlend, C_BLEND, "StandBlend"));
                }
            }

            // V2-specific sections
            if (isV2)
            {
                if (_foldV2Reward = EditorGUILayout.Foldout(_foldV2Reward, "V2 Reward Components", true, EditorStyles.foldoutHeader))
                {
                    DrawGraph(150, window,
                        (m.RawReward, C_RAW, "Raw"),
                        (m.CenteredReward, C_CENTERED, "Centered"),
                        (m.RewardBar, C_BAR, "Bar"));
                    DrawGraph(150, window,
                        (m.Height, C_HEIGHT, "Height"),
                        (m.Orientation, C_ORIENT, "Orient"),
                        (m.ContactReward, C_CONTACT_R, "Contact"),
                        (m.Energy, C_ENERGY, "Energy"),
                        (m.Imitation, C_IMIT, "Imitation"));
                }

                if (_foldV2Progress = EditorGUILayout.Foldout(_foldV2Progress, "Height & State", true, EditorStyles.foldoutHeader))
                {
                    DrawGraph(120, window,
                        (m.HeightFraction, C_PROGRESS, "Height%"),
                        (m.AvgHeightFraction, C_BLEND, "AvgH%"),
                        (m.DiscoveryGate, C_RECOVERY, "DiscGate"),
                        (m.RootZ, C_ROOTZ, "RootZ"));

                    EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
                    Lbl($"Height%: {m.HeightFraction.Latest:F3}", 100);
                    Lbl($"AvgH%: {m.AvgHeightFraction.Latest:F3}", 90);
                    Lbl($"Gate: {m.DiscoveryGate.Latest:F2}", 70);
                    Lbl($"RootZ: {m.RootZ.Latest:F3}", 100);
                    var v2skill = (ContinuousLearningSkillV2)_skill;
                    var v2cfg = v2skill.sacConfig;
                    if (v2cfg.ContextDim > 0)
                    {
                        Lbl($"Ctx: {v2cfg.ContextDim}d", 60);
                        Lbl($"Seq: {v2cfg.ContextSeqLen}", 50);
                    }
                    if (v2cfg.DragForceEnabled)
                        Lbl($"Drag: {v2skill.CurrentDragForce:F0}N", 90);
                    GUILayout.FlexibleSpace();
                    EditorGUILayout.EndHorizontal();
                }

                if (m.DragForce.Count > 0)
                {
                    if (_foldDragForce = EditorGUILayout.Foldout(_foldDragForce, "Drag Force (OU)", true, EditorStyles.foldoutHeader))
                    {
                        DrawGraph(120, window,
                            (m.DragForce, C_DRAG, "DragN"));

                        var v2s = (ContinuousLearningSkillV2)_skill;
                        var cfg = v2s.sacConfig;
                        EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
                        Lbl($"Current: {v2s.CurrentDragForce:F0}N", 100);
                        Lbl($"Range: [{cfg.DragForceMin:F0}, {cfg.DragForceMax:F0}]", 120);
                        Lbl($"Mean: {cfg.DragForceNewtons:F0}N", 80);
                        Lbl($"UB%: {cfg.DragUpperBodyFraction:P0}", 60);
                        GUILayout.FlexibleSpace();
                        EditorGUILayout.EndHorizontal();
                    }
                }

                if (m.WorldModelLoss.Count > 0)
                {
                    if (_foldWorldModel = EditorGUILayout.Foldout(_foldWorldModel, "World Model (Dreaming)", true, EditorStyles.foldoutHeader))
                    {
                        DrawGraph(120, window,
                            (m.WorldModelLoss, C_WMLOSS, "WM Loss"));

                        var sacTrainer = ((ContinuousLearningSkillV2)_skill).Trainer as SACSkillTrainer;
                        if (sacTrainer != null)
                        {
                            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);
                            Lbl($"WM Loss: {sacTrainer.LastWorldModelLoss:F4}", 120);
                            Lbl($"Dreams: {sacTrainer.DreamPhaseCount}", 80);
                            GUILayout.FlexibleSpace();
                            EditorGUILayout.EndHorizontal();
                        }
                    }
                }

                if (_foldTraining = EditorGUILayout.Foldout(_foldTraining, "Training Losses (SAC)", true, EditorStyles.foldoutHeader))
                    DrawGraph(150, window,
                        (m.QLoss, C_QLOSS, "QLoss"),
                        (m.ActorLoss, C_ALOSS, "ActorLoss"),
                        (m.AlphaLoss, C_ALPHA_L, "AlphaLoss"));

                if (_foldAlpha = EditorGUILayout.Foldout(_foldAlpha, "Alpha (Entropy Temperature)", true, EditorStyles.foldoutHeader))
                    DrawGraph(120, window,
                        (m.Alpha, C_ALPHA, "Alpha"));
            }

            // Dynamic metrics from any skill's GetDiagnostics()
            var dynamicMetrics = m.DynamicMetrics;
            if (dynamicMetrics.Count > 0)
            {
                if (_foldDynamic = EditorGUILayout.Foldout(_foldDynamic,
                    $"Skill Metrics ({_skill.Name})", true, EditorStyles.foldoutHeader))
                {
                    var seriesList = new List<(MetricRingBuffer buf, Color col, string label)>();
                    int colorIdx = 0;
                    foreach (var kv in dynamicMetrics)
                    {
                        seriesList.Add((kv.Value,
                            DynamicPalette[colorIdx % DynamicPalette.Length], kv.Key));
                        colorIdx++;
                    }

                    const int MAX_PER_GRAPH = 6;
                    for (int i = 0; i < seriesList.Count; i += MAX_PER_GRAPH)
                    {
                        int count = Math.Min(MAX_PER_GRAPH, seriesList.Count - i);
                        var batch = new (MetricRingBuffer, Color, string)[count];
                        seriesList.CopyTo(i, batch, 0, count);
                        DrawGraph(140, window, batch);
                    }
                }
            }

            // Performance (universal)
            if (_foldPerf = EditorGUILayout.Foldout(_foldPerf, "Performance", true, EditorStyles.foldoutHeader))
            {
                DrawGraph(120, window, (m.TrainingSPS, C_SPS, "Train SPS"));
                DrawGraph(100, window, (m.ReplayCount, C_BUF, "Buffer / Rollout"));
            }

            EditorGUILayout.EndScrollView();
        }

        void DrawToolbar()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            if (_allSkills != null && _allSkills.Length > 1)
            {
                var names = new string[_allSkills.Length];
                for (int i = 0; i < _allSkills.Length; i++)
                    names[i] = $"{_allSkills[i].Name} ({_allSkills[i].gameObject.name})";
                int newIdx = EditorGUILayout.Popup(_selectedSkillIdx, names,
                    EditorStyles.toolbarPopup, GUILayout.Width(200));
                if (newIdx != _selectedSkillIdx)
                {
                    _selectedSkillIdx = newIdx;
                    _skill = _allSkills[newIdx];
                }
            }
            else if (_skill != null)
            {
                EditorGUILayout.LabelField(_skill.Name, EditorStyles.toolbarButton, GUILayout.Width(120));
            }

            GUILayout.FlexibleSpace();

            EditorGUILayout.LabelField("Window:", GUILayout.Width(52));
            _windowIdx = EditorGUILayout.Popup(_windowIdx, WindowLabels,
                EditorStyles.toolbarPopup, GUILayout.Width(70));

            EditorGUILayout.EndHorizontal();
        }

        void DrawSummary(TrainingMetrics m, bool isContinuous, bool isV2 = false)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.helpBox);

            if (isV2)
            {
                float hf = m.HeightFraction.Latest;
                var hfCol = Color.Lerp(PhaseColors[0], PhaseColors[2], hf);
                var oldCol = GUI.contentColor;
                GUI.contentColor = hfCol;
                var v2Skill = (ContinuousLearningSkillV2)_skill;
                int ctxDim = v2Skill.sacConfig.ContextDim;
                string v2Label = ctxDim > 0
                    ? $"V2 h={hf:F2} ctx={ctxDim}"
                    : $"V2 h={hf:F2}";
                EditorGUILayout.LabelField(v2Label, EditorStyles.boldLabel, GUILayout.Width(ctxDim > 0 ? 140 : 80));
                GUI.contentColor = oldCol;
            }
            else if (isContinuous)
            {
                var cls = (ContinuousLearningSkill)_skill;
                var phase = cls.CurrentPhase;
                var phaseCol = PhaseColors[Math.Min((int)phase, PhaseColors.Length - 1)];
                var oldCol = GUI.contentColor;
                GUI.contentColor = phaseCol;
                EditorGUILayout.LabelField(phase.ToString(), EditorStyles.boldLabel, GUILayout.Width(80));
                GUI.contentColor = oldCol;
            }
            else
            {
                EditorGUILayout.LabelField(_skill.Name, EditorStyles.boldLabel, GUILayout.Width(120));
            }

            Lbl($"Raw: {m.RawReward.Latest:F3}", 90);
            if (isContinuous || isV2)
            {
                Lbl($"Ctr: {m.CenteredReward.Latest:F3}", 90);
                Lbl($"\u03B1: {m.Alpha.Latest:F3}", 65);
                Lbl($"QL: {m.QLoss.Latest:F4}", 80);
            }
            Lbl($"SPS: {m.TrainingSPS.Latest:F0}", 65);
            Lbl($"Buf: {m.ReplayCount.Latest:F0}", 80);
            Lbl($"Dec: {_skill.TotalDecisions:N0}", 100);

            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
        }

        static void Lbl(string text, float w)
        {
            EditorGUILayout.LabelField(text, EditorStyles.miniLabel, GUILayout.Width(w));
        }

        void DrawPhaseStrip(TrainingMetrics m, int window)
        {
            Rect rect = GUILayoutUtility.GetRect(position.width - 20, 14);
            if (Event.current.type != EventType.Repaint) return;
            if (m.Phase.Count < 2) return;

            int vis = Math.Min(m.Phase.Count, window);
            int start = m.Phase.Count - vis;
            float pxPerSample = rect.width / vis;

            if (pxPerSample >= 1f)
            {
                for (int i = 0; i < vis; i++)
                {
                    int p = Mathf.Clamp((int)m.Phase[start + i], 0, PhaseColors.Length - 1);
                    float x = rect.x + i * pxPerSample;
                    EditorGUI.DrawRect(new Rect(x, rect.y, Math.Max(1, pxPerSample), rect.height),
                        PhaseColors[p]);
                }
            }
            else
            {
                for (int px = 0; px < (int)rect.width; px++)
                {
                    int sampleIdx = start + (int)((float)px / rect.width * vis);
                    sampleIdx = Math.Min(sampleIdx, m.Phase.Count - 1);
                    int p = Mathf.Clamp((int)m.Phase[sampleIdx], 0, PhaseColors.Length - 1);
                    EditorGUI.DrawRect(new Rect(rect.x + px, rect.y, 1, rect.height),
                        PhaseColors[p]);
                }
            }

            float lx = rect.xMax - 260;
            float ly = rect.y;
            string[] phaseNames = { "Fallen", "Recovering", "Standing", "Moving" };
            for (int i = 0; i < 4; i++)
            {
                EditorGUI.DrawRect(new Rect(lx, ly + 3, 8, 8), PhaseColors[i]);
                var s = new GUIStyle(EditorStyles.miniLabel) { fontSize = 9 };
                GUI.Label(new Rect(lx + 10, ly, 55, 14), phaseNames[i], s);
                lx += 62;
            }
        }

        void DrawGraph(float height, int window,
            params (MetricRingBuffer buf, Color col, string label)[] series)
        {
            Rect rect = GUILayoutUtility.GetRect(position.width - 20, height);
            if (Event.current.type != EventType.Repaint) return;

            EditorGUI.DrawRect(rect, BG_GRAPH);

            const float leftM = 52, rightM = 6, topM = 2, botM = 2;
            Rect plot = new Rect(rect.x + leftM, rect.y + topM,
                rect.width - leftM - rightM, rect.height - topM - botM);

            if (plot.width < 10 || plot.height < 10) return;

            float yMin = float.MaxValue, yMax = float.MinValue;
            foreach (var (buf, _, _) in series)
            {
                if (buf == null || buf.Count == 0) continue;
                int s = Math.Max(0, buf.Count - window);
                for (int i = s; i < buf.Count; i++)
                {
                    float v = buf[i];
                    if (!float.IsNaN(v) && !float.IsInfinity(v))
                    {
                        if (v < yMin) yMin = v;
                        if (v > yMax) yMax = v;
                    }
                }
            }
            if (yMin >= yMax) { yMin -= 0.5f; yMax += 0.5f; }
            float pad = (yMax - yMin) * 0.08f;
            yMin -= pad;
            yMax += pad;
            float yRange = yMax - yMin;

            var gridStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                alignment = TextAnchor.MiddleRight,
                normal = { textColor = new Color(0.5f, 0.5f, 0.5f) },
                fontSize = 9
            };
            for (int g = 0; g <= 4; g++)
            {
                float t = g / 4f;
                float y = plot.yMax - t * plot.height;
                float val = yMin + t * yRange;
                EditorGUI.DrawRect(new Rect(plot.x, y, plot.width, 1), BG_GRID);
                GUI.Label(new Rect(rect.x, y - 6, leftM - 4, 12), FormatVal(val), gridStyle);
            }

            if (yMin < 0 && yMax > 0)
            {
                float zy = plot.yMax - (-yMin / yRange) * plot.height;
                EditorGUI.DrawRect(new Rect(plot.x, zy, plot.width, 1),
                    new Color(0.45f, 0.45f, 0.45f, 0.4f));
            }

            Handles.BeginGUI();
            foreach (var (buf, col, _) in series)
            {
                if (buf == null || buf.Count < 2) continue;
                int vis = Math.Min(buf.Count, window);
                int start = buf.Count - vis;

                int maxPts = Math.Max(2, (int)plot.width);
                int step = Math.Max(1, vis / maxPts);
                int count = 0;
                var pts = new Vector3[(vis / step) + 2];

                for (int i = 0; i < vis; i += step)
                {
                    float v = buf[start + i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) v = 0;
                    float x = plot.x + ((float)i / Math.Max(1, vis - 1)) * plot.width;
                    float y = plot.yMax - ((v - yMin) / yRange) * plot.height;
                    y = Mathf.Clamp(y, plot.y, plot.yMax);
                    pts[count++] = new Vector3(x, y, 0);
                }

                if (count >= 2)
                {
                    if (count < pts.Length) Array.Resize(ref pts, count);
                    Handles.color = col;
                    Handles.DrawAAPolyLine(1.8f, pts);
                }
            }
            Handles.EndGUI();

            float lx = plot.xMax;
            float ly = rect.y + 1;
            for (int i = series.Length - 1; i >= 0; i--)
            {
                var (buf, col, label) = series[i];
                if (buf == null) continue;
                float cur = buf.Count > 0 ? buf.Latest : 0f;
                string txt = $"{label}: {FormatVal(cur)}";
                var content = new GUIContent(txt);
                var sty = new GUIStyle(EditorStyles.miniLabel)
                {
                    normal = { textColor = col },
                    fontSize = 10
                };
                float tw = sty.CalcSize(content).x + 4;
                lx -= tw;
                GUI.Label(new Rect(lx, ly, tw, 13), content, sty);
            }
        }

        static string FormatVal(float v)
        {
            float abs = Mathf.Abs(v);
            if (abs >= 10000) return v.ToString("F0");
            if (abs >= 100) return v.ToString("F1");
            if (abs >= 1) return v.ToString("F2");
            if (abs >= 0.01f) return v.ToString("F3");
            return v.ToString("F4");
        }
    }
}
