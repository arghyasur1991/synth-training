using System;
using System.IO;
using UnityEngine;
using Mujoco;
using static TorchSharp.torch;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Crash-safe persistence for continuous learning state.
    ///
    /// Two-phase commit protocol:
    ///   Phase 1 — Write all files to .tmp (no renames). meta.json.tmp is written
    ///             last as the commit marker. If killed here, all .tmp files are
    ///             incomplete garbage and the previous save is intact.
    ///   Phase 2 — Rename all .tmp → final. If killed here, recovery on next load
    ///             finishes any remaining renames (meta.json.tmp guarantees all
    ///             data .tmp files are complete).
    /// </summary>
    public class StatePersister
    {
        private const string META_FILE = "meta.json";
        private const string BUFFER_FILE = "replay_buffer.bin";
        private const string NORMALIZER_FILE = "normalizer.bin";
        private const string REWARD_FILE = "reward_state.bin";
        private const string CURRICULUM_FILE = "curriculum_state.bin";
        private const string PHYSICS_FILE = "physics_state.bin";
        private const string AGENT_DIR = "agent";
        private const int IO_BUFFER_SIZE = 1024 * 1024;

        private readonly string _directory;
        private int _loadedDecisionCount;

        public string Directory => _directory;
        public int LoadedDecisionCount => _loadedDecisionCount;

        public StatePersister(string directory)
        {
            _directory = directory;
        }

        public bool HasSavedState()
        {
            return File.Exists(Path.Combine(_directory, META_FILE));
        }

        public void DeleteAll()
        {
            if (Directory.Exists(_directory))
            {
                Directory.Delete(_directory, recursive: true);
                Debug.Log($"StatePersister: Deleted save directory {_directory}");
            }
        }

        public unsafe void Save(SACAgent agent, ReplayBuffer buffer,
            ObservationNormalizer normalizer, ContinuingReward reward,
            int totalDecisions, MujocoLib.mjData_* physicsData = null,
            ActionCurriculum curriculum = null)
        {
            Directory.CreateDirectory(_directory);

            // Phase 1: write all .tmp files (no renames)
            WriteAgentTmp(agent);
            WriteBinaryTmp(Path.Combine(_directory, BUFFER_FILE), IO_BUFFER_SIZE,
                bw => buffer.Save(bw));
            WriteBinaryTmp(Path.Combine(_directory, NORMALIZER_FILE), 0,
                bw => normalizer.Save(bw));
            WriteBinaryTmp(Path.Combine(_directory, REWARD_FILE), 0,
                bw => reward.Save(bw));
            if (curriculum != null)
                WriteBinaryTmp(Path.Combine(_directory, CURRICULUM_FILE), 0,
                    bw => curriculum.Save(bw));
            if (physicsData != null)
                WritePhysicsStateTmp(physicsData);
            WriteMetaTmp(agent, buffer, totalDecisions); // commit marker — last

            // Phase 2: rename all .tmp → final
            PromoteAllTmpFiles();
        }

        public void SaveWithSnapshot(SACAgent agent, ReplayBuffer buffer,
            ObservationNormalizer normalizer, ContinuingReward reward,
            int totalDecisions, double[] qpos, double[] qvel, double[] ctrl,
            ActionCurriculum curriculum = null)
        {
            Directory.CreateDirectory(_directory);

            // Phase 1: write all .tmp files (no renames)
            WriteAgentTmp(agent);
            WriteBinaryTmp(Path.Combine(_directory, BUFFER_FILE), IO_BUFFER_SIZE,
                bw => buffer.Save(bw));
            WriteBinaryTmp(Path.Combine(_directory, NORMALIZER_FILE), 0,
                bw => normalizer.Save(bw));
            WriteBinaryTmp(Path.Combine(_directory, REWARD_FILE), 0,
                bw => reward.Save(bw));
            if (curriculum != null)
                WriteBinaryTmp(Path.Combine(_directory, CURRICULUM_FILE), 0,
                    bw => curriculum.Save(bw));
            if (qpos != null)
                WritePhysicsSnapshotTmp(qpos, qvel, ctrl);
            WriteMetaTmp(agent, buffer, totalDecisions); // commit marker — last

            // Phase 2: rename all .tmp → final
            PromoteAllTmpFiles();
        }

        // ─── Phase 1: Write .tmp files ───────────────────────────────

        private static void WriteBinaryTmp(string finalPath, int bufferSize,
            Action<BinaryWriter> writeAction)
        {
            string tmpPath = finalPath + ".tmp";
            using var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write,
                FileShare.None, bufferSize > 0 ? bufferSize : 4096);

            using Stream writeStream = bufferSize > 0
                ? new BufferedStream(fs, bufferSize)
                : (Stream)fs;

            using (var bw = new BinaryWriter(writeStream, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                writeAction(bw);
            }

            writeStream.Flush();
            fs.Flush(true);
        }

        private void WriteAgentTmp(SACAgent agent)
        {
            string tmpDir = Path.Combine(_directory, AGENT_DIR + "_tmp");
            if (Directory.Exists(tmpDir))
                Directory.Delete(tmpDir, true);
            using var scope = NewDisposeScope();
            agent.Save(tmpDir);
        }

        private void WriteMetaTmp(SACAgent agent, ReplayBuffer buffer, int totalDecisions)
        {
            var meta = new LearningMetadata
            {
                totalDecisions = totalDecisions,
                trainSteps = agent.TrainSteps,
                alpha = agent.Alpha,
                replayCount = buffer.Count,
                timestamp = DateTime.UtcNow.ToString("o"),
                version = 1
            };
            string tmpPath = Path.Combine(_directory, META_FILE + ".tmp");
            File.WriteAllText(tmpPath, JsonUtility.ToJson(meta, prettyPrint: true));
        }

        private unsafe void WritePhysicsStateTmp(MujocoLib.mjData_* data)
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null) return;
            var model = MjScene.Instance.Model;
            int nq = (int)model->nq;
            int nv = (int)model->nv;
            int nu = (int)model->nu;

            WriteBinaryTmp(Path.Combine(_directory, PHYSICS_FILE), 0, bw =>
            {
                bw.Write(nq);
                bw.Write(nv);
                bw.Write(nu);
                for (int i = 0; i < nq; i++) bw.Write(data->qpos[i]);
                for (int i = 0; i < nv; i++) bw.Write(data->qvel[i]);
                for (int i = 0; i < nu; i++) bw.Write(data->ctrl[i]);
            });
        }

        private void WritePhysicsSnapshotTmp(double[] qpos, double[] qvel, double[] ctrl)
        {
            WriteBinaryTmp(Path.Combine(_directory, PHYSICS_FILE), 0, bw =>
            {
                bw.Write(qpos.Length);
                bw.Write(qvel?.Length ?? 0);
                bw.Write(ctrl?.Length ?? 0);
                for (int i = 0; i < qpos.Length; i++) bw.Write(qpos[i]);
                if (qvel != null)
                    for (int i = 0; i < qvel.Length; i++) bw.Write(qvel[i]);
                if (ctrl != null)
                    for (int i = 0; i < ctrl.Length; i++) bw.Write(ctrl[i]);
            });
        }

        // ─── Phase 2: Promote .tmp → final ──────────────────────────

        private void PromoteAllTmpFiles()
        {
            // Promote agent directory first
            string agentDir = Path.Combine(_directory, AGENT_DIR);
            string agentTmpDir = agentDir + "_tmp";
            if (Directory.Exists(agentTmpDir))
            {
                Directory.CreateDirectory(agentDir);
                foreach (var tmpFile in Directory.GetFiles(agentTmpDir))
                {
                    string finalPath = Path.Combine(agentDir, Path.GetFileName(tmpFile));
                    ReplaceFile(tmpFile, finalPath);
                }
                Directory.Delete(agentTmpDir, true);
            }

            // Promote all .tmp files in the save directory
            foreach (var tmpPath in Directory.GetFiles(_directory, "*.tmp"))
            {
                string finalPath = tmpPath.Substring(0, tmpPath.Length - 4);
                ReplaceFile(tmpPath, finalPath);
            }
        }

        private static void ReplaceFile(string tmpPath, string finalPath)
        {
            if (File.Exists(finalPath))
                File.Delete(finalPath);
            File.Move(tmpPath, finalPath);
        }

        // ─── Load with recovery ─────────────────────────────────────

        public void Load(SACAgent agent, ReplayBuffer buffer,
            ObservationNormalizer normalizer, ContinuingReward reward,
            ActionCurriculum curriculum = null)
        {
            RecoverFromInterruptedSave();

            string metaPath = Path.Combine(_directory, META_FILE);
            if (!File.Exists(metaPath))
                throw new FileNotFoundException("No saved state found", metaPath);

            var meta = JsonUtility.FromJson<LearningMetadata>(File.ReadAllText(metaPath));
            _loadedDecisionCount = meta.totalDecisions;

            string agentDir = Path.Combine(_directory, AGENT_DIR);
            if (Directory.Exists(agentDir))
                agent.Load(agentDir);

            string bufferPath = Path.Combine(_directory, BUFFER_FILE);
            if (File.Exists(bufferPath))
            {
                using var br = new BinaryReader(
                    new BufferedStream(File.OpenRead(bufferPath), IO_BUFFER_SIZE));
                buffer.Load(br);
            }

            string normPath = Path.Combine(_directory, NORMALIZER_FILE);
            if (File.Exists(normPath))
            {
                using var br = new BinaryReader(File.OpenRead(normPath));
                normalizer.Load(br);
            }

            string rewardPath = Path.Combine(_directory, REWARD_FILE);
            if (File.Exists(rewardPath))
            {
                using var br = new BinaryReader(File.OpenRead(rewardPath));
                reward.Load(br);
            }

            string curriculumPath = Path.Combine(_directory, CURRICULUM_FILE);
            if (curriculum != null && File.Exists(curriculumPath))
            {
                try
                {
                    using var br = new BinaryReader(File.OpenRead(curriculumPath));
                    curriculum.Load(br);
                    Debug.Log($"StatePersister: Loaded curriculum state — stage {curriculum.CurrentStage}");
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"StatePersister: Curriculum load failed ({e.Message}), starting from stage 0");
                }
            }

            Debug.Log($"StatePersister: Loaded state from {_directory} — " +
                      $"decisions={meta.totalDecisions}, train_steps={meta.trainSteps}, " +
                      $"alpha={meta.alpha:F4}, replay={meta.replayCount}, " +
                      $"saved={meta.timestamp}");
        }

        /// <summary>
        /// Two-phase commit recovery:
        ///   - meta.json.tmp exists → Phase 1 completed, Phase 2 was interrupted.
        ///     All .tmp files are complete. Promote them (finish Phase 2).
        ///   - meta.json.tmp absent but other .tmp exist → Phase 1 was interrupted.
        ///     Data is incomplete. Delete all .tmp files, keep previous save.
        /// </summary>
        private void RecoverFromInterruptedSave()
        {
            if (!Directory.Exists(_directory)) return;

            bool commitMarkerExists = File.Exists(Path.Combine(_directory, META_FILE + ".tmp"));

            if (commitMarkerExists)
            {
                Debug.Log("StatePersister: Recovering interrupted save — promoting .tmp files");
                PromoteAllTmpFiles();
            }
            else
            {
                // Delete any orphaned .tmp files (incomplete Phase 1)
                foreach (var tmpPath in Directory.GetFiles(_directory, "*.tmp", SearchOption.AllDirectories))
                {
                    File.Delete(tmpPath);
                }

                string agentTmpDir = Path.Combine(_directory, AGENT_DIR + "_tmp");
                if (Directory.Exists(agentTmpDir))
                    Directory.Delete(agentTmpDir, true);
            }
        }

        public unsafe bool LoadPhysicsState(MujocoLib.mjData_* data)
        {
            string path = Path.Combine(_directory, PHYSICS_FILE);
            if (!File.Exists(path)) return false;
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null) return false;
            var model = MjScene.Instance.Model;

            using var br = new BinaryReader(File.OpenRead(path));
            int savedNq = br.ReadInt32();
            int savedNv = br.ReadInt32();
            int savedNu = br.ReadInt32();

            int readNq = Math.Min(savedNq, (int)model->nq);
            for (int i = 0; i < readNq; i++)
                data->qpos[i] = br.ReadDouble();
            for (int i = readNq; i < savedNq; i++)
                br.ReadDouble();

            int readNv = Math.Min(savedNv, (int)model->nv);
            for (int i = 0; i < readNv; i++)
                data->qvel[i] = br.ReadDouble();
            for (int i = readNv; i < savedNv; i++)
                br.ReadDouble();

            int readNu = Math.Min(savedNu, (int)model->nu);
            for (int i = 0; i < readNu; i++)
                data->ctrl[i] = br.ReadDouble();
            for (int i = readNu; i < savedNu; i++)
                br.ReadDouble();

            return true;
        }
    }

    [Serializable]
    public class LearningMetadata
    {
        public int totalDecisions;
        public int trainSteps;
        public float alpha;
        public int replayCount;
        public string timestamp;
        public int version;
    }
}
