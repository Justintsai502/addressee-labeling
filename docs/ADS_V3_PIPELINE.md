# ADS v3 標註 pipeline（clause-level addressee → 12.5 Hz frame labels）

把 ttv2 這類 4 人 TTS 對話（每個座位一條音軌 + `script.jsonl`）轉成每個 agent 座位的 12.5 Hz ADS 標籤。
LLM 以子句為單位判斷「對誰說」，用 5 票的比例當 soft label；時間由 Parakeet 的 word timestamps 對到音軌。

- 設計與決定的來源：mpmoshi `artifacts/ads_v3_labeling_1011/TODO.md`（rev 3，含兩輪 review）、`docs/ADS_LABEL_SPEC.md` v3。
- 第一次實跑：mpmoshi EXP-062（20 段 pilot 對話，`artifacts/ads_v3_labeling_1011/RESULTS_dev20.md`）。
- 所有標籤都是 LLM 產生，**沒有人工審核**（`human_audited=false`）。

---

## 1. 標籤定義

每個 frame、每個 agent 座位一個類別：

| 類別 | 意思 |
|---|---|
| `none` | 沒有人在說話 |
| `self` | agent 座位自己在說話 |
| `to_agent_itself` | 有人直接對 agent 說（agent 單獨，或 agent + 部分人；**不是**「只有 agent」） |
| `to_everyone` | 對除了說話者以外的所有人說（包含 agent） |
| `not_to_agent` | 對其他人說，不包含 agent（也包含「對所有人類但不含 agent」） |

沒有 `uncertain` 類別。模糊的程度用 soft label 表示：5 票裡有幾票投給哪一類。

LLM 先對每個子句給出「說話對象的座位集合」，再換算成上表。`GROUP`、或等於「所有其他人」的集合都視為 `ALL`。
每段對話會產生 4 個 agent view（每個座位輪流當 agent）。**同一段對話的 4 個 view、同一個 `source_folder`
的對話，切 train/test 時必須放在同一邊。**

## 2. 流程

| 步驟 | 程式 | 硬體 | 做什麼 |
|---|---|---|---|
| 1 prepare | `scripts/00c_prepare_ttv2.py` | CPU | ttv2 目錄 → conversations JSONL（用顯示名稱當 speaker） |
| 2 label | `scripts/11_clause_label.py` | GPU（vLLM） | 子句切分 → 每個子句 5 票（causal：只看之前的 turn 和本句到該子句為止）→ 無效票重抽一次 |
| 3 align | `scripts/12_align_ttv2.py` | GPU（NeMo） | 每個座位的整條音軌跑 Parakeet，取 word timestamps |
| 4 frames | `scripts/13_build_frames_v5.py` | CPU | VAD（整條個人音軌）+ 子句時間段 → 12.5 Hz soft labels |
| 5 report | `scripts/14_report_v5.py` | CPU | 統計與品質檢查 |

核心邏輯在 `src/clause_v5.py`（子句切分、prompt、投票解析、類別換算）和 `src/frames_v5.py`（VAD、對齊、frame 規則）。

## 3. 環境

- **vLLM 環境**：能跑 `Qwen/Qwen3-32B`（bf16 約 64 GB，一張 80 GB GPU 即可）。我們用 vLLM 0.26.0、
  模型 revision `9216db5781bf21249d130ec9da846c4624c16137`。離線節點請先下載權重並設 `HF_HOME`、`HF_HUB_OFFLINE=1`。
- **NeMo 環境**：`nemo_toolkit[asr]`、`librosa`、`soundfile`、`numpy`。ASR 模型 `nvidia/parakeet-tdt-0.6b-v2`
  （可給 HF id、`.nemo` 檔或含 `.nemo` 的資料夾）。步驟 4 也用這個環境（需要 numpy + soundfile）。
- **CPU python**：步驟 1、5 只用標準函式庫。

## 4. 資料格式

`<TTV2_ROOT>/<dialogue_id>/` 底下需要：
- `script.jsonl`：每行一個 utterance，含 `speaker`（座位 A–D）、`start`、`end`、`text`、`addressing`、
  `function`，第一行要有 `display_names`（或另有 `names.json`）。
- `meta.json`：`setup.source_folder` 用來分組。
- 每個座位一條音軌，預設檔名 `audio_qwen3-tts-1.7b-clone_{seat}.wav`（可用 `AUDIO_PATTERN` 改）。

## 5. 一鍵執行

```bash
cd addressee-labeling
TTV2_ROOT=/path/to/turn_taking_2000_v2 \
OUT=outputs/ads_v3_v5/run_$(date +%m%d) \
IDS_FILE=my_ids.txt \                  # 或 ALL=1，或 CONVS=<已有的 conversations JSONL>
PY_VLLM=/path/to/vllm-env/bin/python \
PY_NEMO=/path/to/nemo-env/bin/python \
HF_HOME=/path/to/hf_cache \
sbatch -A <ACCOUNT> --job-name=adsv3 --time=04:00:00 scripts/run_ads_v3.sh
```

可選變數：`MODEL`、`ASR_MODEL`、`K`（5）、`SEED`（20261011）、`FIXED=data/ttv2/fixed_cases_v5.json`（固定邊界案例）、
`GREEDY=1`、`FULL=1`（多跑一次看得到後文的對照版，只做比較用）、`LOCK`（品質門檻檔，見 mpmoshi
`artifacts/ads_v3_labeling_1011/LOCK_dev20.json`）、`STEPS="label frames"`（只跑部分步驟）。

**中斷後重送同一個指令即可**：標註會跳過已完成的對話（`--resume`，每 `--chunk-convs 50` 段寫一次檔），
對齊會跳過已有輸出的對話。

不用 GPU 先試流程（隨機答案）：
```bash
TTV2_ROOT=... OUT=/tmp/ads_dry IDS_FILE=my_ids.txt PY_VLLM=python3 PY_NEMO=/path/to/nemo-env/bin/python \
BACKEND=mock STEPS="prepare label" bash scripts/run_ads_v3.sh
```

## 6. 分步執行

```bash
python3 scripts/00c_prepare_ttv2.py --root $TTV2_ROOT --ids-file my_ids.txt \
    --out-convs $OUT/convs.jsonl --out-ref $OUT/generator_ref.jsonl
$PY_VLLM scripts/11_clause_label.py --convs $OUT/convs.jsonl --mode causal --out-dir $OUT/causal --resume
$PY_NEMO scripts/12_align_ttv2.py --root $TTV2_ROOT --convs $OUT/convs.jsonl --out-dir $OUT/align
$PY_NEMO scripts/13_build_frames_v5.py --convs $OUT/convs.jsonl --votes $OUT/causal/clause_votes.jsonl \
    --align-dir $OUT/align --root $TTV2_ROOT --out-dir $OUT/frames
python3 scripts/14_report_v5.py --causal $OUT/causal --frames $OUT/frames --out $OUT/report.json
```

## 7. 輸出

**`frames/<dialogue_id>.npz`**（訓練要用的就是這個）

| key | shape | 內容 |
|---|---|---|
| `activity` | `[4, T]` uint8 | 每個座位每個 frame 有沒有聲音（座位順序 = 排序後的 A, B, C, D） |
| `soft_<seat>` | `[T, 5]` float32 | 該座位當 agent 時的 soft label，欄位順序 `none, self, to_agent_itself, to_everyone, not_to_agent` |
| `mask_<seat>` | `[T]` uint8 | 1 = 算 loss，0 = 遮罩 |
| `reason_<seat>` | `[T]` uint8 | 該 frame 的判定原因，索引見 `13_build_frames_v5.py` 的 `REASONS` |

frame `i` 對應時間 `[i·0.08, (i+1)·0.08)` 秒，從音檔開頭算。

```python
import numpy as np
z = np.load("frames/tr1670.npz")
soft, mask = z["soft_A"], z["mask_A"]          # 座位 A 當 agent
hard = np.where(mask == 1, soft.argmax(1), -1)  # -1 = 遮罩
```

**`frames/clauses.jsonl`**：每個子句一行，含時間段 `t0/t1`、對齊品質、`function`、`source_folder`、多數票、
4 個 agent view 的類別與 soft label、每票的 cue 診斷。
**`causal/clause_votes.jsonl`**：每個子句的 5 個 slot（原始回答、解析結果、重抽紀錄）、`valid_vote_count`、多數票。
**`report.json`**：無效票率、遮罩率、投票分歧、對齊品質、frame 原因分布等。

## 8. Frame 規則

依序判斷每個 frame（agent 座位 a）：
1. 只有 a 有聲音 → `self`
2. 沒人有聲音 → `none`（子句內的停頓也算）
3. 一個人有聲音、落在他某個子句的時間段內（覆蓋該 frame ≥ 50%，同分取較早的）→ 該子句的 soft label
4. 有聲音但沒有可靠的時間段 → 遮罩（不會變成 `none`）
5. 多人同時有聲音 → 跟著**最晚開始說話的人**（新進來的人）：是 a → `self`，是別人 → 套用 3/4；
   同一個 frame 同時開口 → 遮罩

另外遮罩：有效票少於 3 張的子句；script 裡含 `[to X]` 舞台指示的 turn（TTS 會把它念出來）；
說話者叫自己名字的 turn（生成錯誤，見 [`TTV2_DATA_ISSUES.md`](TTV2_DATA_ISSUES.md)）。以上都只遮罩其他座位的 view，
說話者自己的座位仍是 `self`。

前處理（`00c`）會把座位字母呼喚 `[C],` 從文字中拿掉、其他位置的 `[C]` 換成 "someone"，因為音檔裡聽不到名字
（`--keep-seat-letters` 關掉；EXP-062 dev20 是舊行為）。原始文字保留在每個 turn 的 `script_text`。

## 9. 成本（dev20 實測，1 張 GPU）

20 段對話（360 turn、535 子句）：causal 標註 + greedy 14 分鐘、看全文對照版 11 分鐘、對齊 1 分鐘。
大約每 100 段對話 1 GPU 小時（只跑 causal），實際取決於 GPU 型號與 thinking 長度。

## 10. 已知限制

- 2026-10-12 在全新 clone 上模擬實跑（3 段對話，1 張 GPU，17 分鐘，含看全文對照版與 ASR 檢查）：
  門檻全部通過、續跑會跳過已完成的部分。測試用的 ASR 是給本機路徑；用 HF id 自動下載沒有在離線節點上試過。

- VAD 用能量門檻（RMS > 1e-3），前提是個人音軌在沒說話時是數位靜音（TTS 資料成立）；真實錄音不適用。
- script 的時間和實際音檔最多差約 1.5 秒，所以時間一律以音檔為準（VAD + ASR），不用 script 的 `start/end`。
- 子句切分是規則式（標點 + 呼喚語 + 至少 3 個字）；短的前置從屬子句（"If we focus,"）會自成一個子句。
- `cue`（LLM 說的決定性字詞）5 票只有 22% 一致，只能當診斷，不能當「從哪裡開始可判定」的標籤。
- 標籤是子句結束後才判定的語意標籤；子句開頭的 frame 不一定已經聽得出對象。評估時間點請用子句結尾。
- dev20 不是 held-out 資料，EXP-062 的數字只代表流程與規則一致性，不代表標註正確率。

## 11. 資料問題

ttv2 的生成錯誤（自己叫自己、座位字母、開場句標錯等）整理在 [`TTV2_DATA_ISSUES.md`](TTV2_DATA_ISSUES.md)，
清單 `data/ttv2/script_issues.csv`，用 `scripts/15_audit_ttv2_script.py` 重新產生。

## 12. 測試

```bash
python3 tests/test_clause_v5.py   # 57 項：子句切分、prompt 因果性、投票解析、類別換算、遮罩、frame 規則、對齊 fallback
python3 tests/test_pipeline.py    # 原有測試
```

Seed 規則：每個 request 的 seed = sha256(base seed | 對話 | turn | 子句 | 輪次 | slot)。
（EXP-062 的 dev20 實跑用的是舊規則 `seed + 10·index + slot`，超過一萬個子句會撞號，2026-10-12 已改。）
