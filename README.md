# Addressee Labeling — TODO 1 驗證框架

> Multi-Speaker Full Duplex 專案，投影片第 18 頁 TODO 第 1 項：
> **「Label current datasets the new Addressee information.」**

目標：為現有的多人對話資料集（AMI、PersonaPlex 2000hrs…）標上 **addressee**（`<ads>`）
資訊——也就是「這一段話是講給誰聽的」。訓練階段二（投影片 17）要讓模型學會產生
`<ads>` token，所以得先有 addressee 標註。

本 repo 實作你提出的驗證策略：
**先用強模型（Gemini 3 Pro，audio + transcript）建一組 golden set，再檢查只吃 transcript
的 Qwen 能不能貼近 golden set。** 若夠接近，就用便宜、可自架的 Qwen 去標整個 2000 小時語料。

---

## 1. 這個方法好不好？（評估）

**結論：方法本身是對的，我認為可行，但要補幾個關鍵設計才站得住腳。**

### 為什麼這個方向是對的

| 優點 | 說明 |
|---|---|
| 標準且正確的方法論 | 「用最強模型 + 最多資訊建參考標準，再驗證便宜 labeler」是 model-based annotation 的標準做法。 |
| 成本正確 | Gemini 吃 audio token 很貴，跑滿 2000hr 不切實際；Qwen 可**自架、零邊際成本**。先小規模驗證再放大是對的。 |
| golden 用對工具 | addressee 有些線索（語氣、轉向誰講、是否針對個人）只在 audio 裡，Gemini 多模態 + audio+transcript 拿到最多資訊。 |
| 與現有 pipeline 一致 | 你們 persona metadata（投影片 19）已用 Qwen3-32B，addressee 沿用同一套 infra。 |

### 必須補強的地方（否則數字沒意義）

1. **「golden」不等於 ground truth。** Gemini 的標註仍是模型預測。**務必再抽 100–200 turn 做
   人工校對**，用來 (a) 量 Gemini 本身的準確率、(b) 當真正的 gold。建議三層：
   `human gold ⊂ Gemini silver ⊂ Qwen bulk`。
2. **先定義 label schema 與 guideline**（見 §2），再開始標。沒有明確定義，agreement 數字無意義。
   這是**最重要的前置步驟**。
3. **指標要用 chance-corrected。** addressee 類別極度不平衡（兩人對話時 addressee 幾乎必然是對方），
   純 accuracy 會虛高。必須報 **Cohen's kappa** + **per-class F1** + **confusion matrix**（本框架都做了）。
4. **Speaker id 要對齊。** Gemini 和 Qwen 必須用**同一組 speaker id**（來自 diarization），
   否則兩邊的 addressee 無法比較。本框架強制兩個 labeler 讀同一份 `render_window` 產生的文字。
5. **先寫死「夠接近」的門檻**（pre-register），例如 overall kappa ≥ 0.75、exact-match ≥ 0.80，
   **看結果前就決定**，避免事後搬龍門。
6. **抽樣要 stratify。** golden set 若只抽到簡單的兩人 turn，agreement 會假性偏高。
   要按「同時說話人數」「是否 overlap」分層抽樣，確保難的 case 有足夠樣本。
7. **加一條 ablation：Gemini(text-only) vs Gemini(audio+text)。** 這樣能把
   「Qwen 比 Gemini 弱」和「純文字比多模態少了資訊」兩件事拆開——直接量出 audio 到底值多少。
   （`scripts/01_build_golden.py --no-audio` 就是這條。）
8. **你自己在投影片 15 點出的最難 case**：兩人同時講話時很難標。overlap turn 的 addressee 從文字最難拿，
   audio 幫助最大——本框架把 overlap turn 單獨拉出來報。

> 一句話：**方向對、值得做**；把上面 1–8 補上，這個驗證就能真正回答「Qwen 能不能取代 Gemini 標 addressee」。

---

## 2. Label schema（標註定義）

- **標註單位**：一個 diarized **turn**（之後在 data-prep 再展開成 frame-level 的 `<ads>` token stream）。
- **一則 turn 的 addressee 是一個「集合」**（可能同時對多人），取值來自：
  - 本對話中**其他** speaker 的 id（不含說話者自己）；
  - `GROUP`：對全場 / 所有人講；
  - `UNKNOWN`：真的判斷不出來。
- `UNKNOWN` 是互斥的：不確定就只給 `UNKNOWN`，不要跟具體 id 混用。
- **判斷線索**（golden 與 candidate 用**完全相同**的 guideline，唯一差別是有沒有 audio）：
  vocative（叫名字）、問答相鄰（誰回話）、第二人稱單/複數、話題延續、backchannel 對象是當前 floor-holder、
  兩人對話時就是對方。完整 prompt 見 [`src/prompts.py`](src/prompts.py)。

> 關於 agent：AMI / PersonaPlex 是人對人語料，addressee 就在人類 speaker 之間。等把 Moshi agent 當成其中一個
> persona 插進訓練資料時，「addressee = agent」只是「addressee = agent 扮演的那個 speaker」的特例——
> 在標註階段不需要特別處理。

---

## 3. Pipeline

```
資料 (jsonl, 每行一段對話 + audio_path)
        │
        ├──①  scripts/01_build_golden.py   Gemini 3 Pro (audio+transcript)  ─►  outputs/golden.jsonl
        │
        ├──②  scripts/02_run_candidate.py  Qwen3-32B   (transcript only)     ─►  outputs/candidate.jsonl
        │
        └──③  scripts/03_evaluate.py       純離線比對                         ─►  報表 + PASS/FAIL
```

① 和 ② **在 server 上跑**（要下載/呼叫大模型）；③ 在哪都能跑（純 Python，無外部套件）。

長對話會自動切 window（`max_turns_per_window` + 前文 `context_turns`），避免爆 context，
兩個 labeler 用同一套切法。

---

## 4. 評估指標（`scripts/03_evaluate.py`）

| 指標 | 為什麼要它 |
|---|---|
| **exact-set match** | turn 的 addressee 集合完全相同的比例。 |
| **mean Jaccard** | 多 addressee 時的部分分數（例如 gold `{Bob,Carol}` vs pred `{Bob}`）。 |
| **Cohen's kappa** | 在「單一 addressee」子集上做 chance-corrected agreement，修正類別不平衡。 |
| **per-class F1（one-vs-rest）** | 看每個 addressee 類別（含 GROUP/UNKNOWN）各自的 precision/recall。 |
| **分層（strata）** | overall / 兩人 / 多人(≥3) / overlap / backchannel 分開報——**難的 case 才是重點**。 |
| **confusion matrix** | 看 Qwen 常把誰錯認成誰（例如 GROUP↔特定人）。 |

**Acceptance**：`--accept-kappa`、`--accept-exact`（預設 0.75 / 0.80）。**跑之前先定好、寫進 `config.yaml`。**

---

## 5. 怎麼跑

### 現在就能跑（離線，無需任何套件、無需大模型）

```bash
cd addressee-labeling
python3 tests/test_pipeline.py     # 單元測試（指標 / 解析 / 切窗）
python3 run_demo.py                # 用 mock labeler 跑完整 pipeline
```

**「扮演模型」的實跑驗證**（本 repo 已附）：`outputs/golden.jsonl` 與 `outputs/candidate.jsonl`
是我人工扮演兩個模型（golden 用 audio+transcript 推理、candidate 只用 transcript）標出來的，
再用真正的 `scripts/03_evaluate.py` 比對，重現：

```bash
python3 scripts/03_evaluate.py \
  --conversations data/sample/conversations.jsonl \
  --gold outputs/golden.jsonl --pred outputs/candidate.jsonl \
  --accept-kappa 0.75 --accept-exact 0.80 --report outputs/report.json
```

得到 overall exact 0.85 / kappa 0.863（PASS），而且**分歧剛好集中在 overlap turn
（exact 0.50）、bare backchannel、以及 "both" 多 addressee**——正是理論上 audio 才幫得上的地方。
這證明：pipeline 通、指標會把「audio 有沒有差」定位到正確的 case。

### 之後在 server 上跑（真的大模型）

```bash
pip install -r requirements.txt          # google-genai, openai, pyyaml

# ⓪ 準備真資料（AMI：下載音檔+標註→轉成 conversations.jsonl；資料大且有授權，已 gitignore）
bash scripts/download_ami.sh ES2002a ES2002b     # 產生 data/ami/conversations.jsonl

# ① golden：Gemini（Google API，不需下載權重，但 audio token 要錢 → 只跑抽樣子集）
#    Gemini labeler 會照 window 把音檔切片再上傳（slice_audio 預設開），長會議不會重送整段。
#    注意：Pro 級模型 free tier 額度是 0（實測 429 RESOURCE_EXHAUSTED, limit:0），
#    要在 AI Studio 該 key 的 project 點「Set up billing」開通計費才能用 Pro；
#    切片後單次測試(30 turns/~4min)花費很低。billing 開通前可先用免費的
#    --model gemini-2.5-flash 測通其餘流程（切片/上傳/parse），但品質不能當正式 golden。
export GEMINI_API_KEY=...
python3 scripts/01_build_golden.py --conversations data/ami/conversations.jsonl --out outputs/golden.jsonl

# ② candidate：下載模型、in-process 載入跑（--backend local，換模型只要換 --model）
python3 scripts/02_run_candidate.py --backend local --model Qwen/Qwen3-32B \
    --conversations data/ami/conversations.jsonl --out outputs/candidate.jsonl
#   想換模型比較，改 --model 即可（權重會下載到 ~/.cache/huggingface）：
#     --model Qwen/Qwen3-8B
#     --model meta-llama/Llama-3.1-8B-Instruct
#     --engine hf            # vllm 載不動某些模型時的萬用後備
#   多卡：--tensor-parallel-size 2
#   （或 --backend endpoint 走 vllm serve / DashScope 的 HTTP endpoint）

# ③ 比對
python3 scripts/03_evaluate.py --conversations data/golden_sample.jsonl \
    --gold outputs/golden.jsonl --pred outputs/candidate.jsonl --report outputs/report.json
```

> **關於「不要實作下載大模型」**：本 repo 的程式碼**不含任何下載 / 載入模型權重的邏輯**。
> Gemini 走 Google API；Qwen 走 **OpenAI-相容 endpoint**（你在 server 上 `vllm serve` 或指到 DashScope），
> 程式只當 client。所以在你這台電腦 import / 建構 labeler 完全不會去抓模型（已驗證：未安裝
> google-genai / openai 時，整個專案仍可 import 並跑完離線測試）。

---

## 6. 專案結構

```
addressee-labeling/
├── README.md                     ← 本報告
├── requirements.txt              ← 離線核心零依賴；大模型套件僅 server 需要
├── config.example.yaml           ← server 跑真模型用；門檻寫這裡
├── run_demo.py                   ← 離線 end-to-end demo（mock labeler）
├── src/
│   ├── schema.py                 ← Conversation / Turn / AddresseeLabel + label 定義
│   ├── io_utils.py               ← jsonl 讀寫
│   ├── transcript_format.py      ← 把對話 render 成模型讀的文字 + 切 window（兩 labeler 共用）
│   ├── prompts.py                ← golden / candidate 的 system prompt（除 audio 外完全相同）
│   ├── parsing.py                ← 穩健解析模型輸出 + 驗證 addressee 合法性
│   ├── evaluate.py               ← 全部指標（純 Python，無依賴）
│   ├── pipeline.py               ← 跑 labeler + 存檔
│   ├── audio_utils.py            ← ffmpeg 音檔切片（golden 按 window 切，長會議不重送）
│   ├── config.py                 ← YAML 設定（${ENV} 展開）
│   └── labelers/
│       ├── base.py               ← AddresseeLabeler 抽象類 + window template method
│       ├── gemini_labeler.py     ← Gemini 3 Pro，audio+transcript，按 window 切片（server）
│       ├── qwen_labeler.py       ← candidate，OpenAI-相容 endpoint（vllm serve / DashScope）
│       ├── local_labeler.py      ← candidate，下載權重 in-process 載入（vllm/hf，換模型用）
│       └── mock_labeler.py       ← 離線 heuristic（demo / 測試用，無 LLM）
├── scripts/
│   ├── download_ami.sh           ← 下載 AMI 音檔+標註並轉檔
│   ├── 00_prepare_ami.py         ← AMI NXT XML → conversations.jsonl
│   └── 01_build_golden / 02_run_candidate / 03_evaluate
├── data/sample/conversations.jsonl  ← 3 段合成對話（dyadic/triad/quad，含難 case）
├── data/ami/                     ← 真 AMI 資料（gitignored，用 download_ami.sh 重建）
├── outputs/                      ← 「扮演模型」的 golden/candidate 標註 + 評估報表
└── tests/test_pipeline.py        ← 純 Python 單元測試
```

---

## 7. 建議的後續步驟

1. **先定 schema + 標 50–100 turn 的人工 micro-gold**，量 Gemini(audio+text) 自己的準確率。
2. **stratified 抽樣** golden set（依 speaker 數 / overlap），跑 ①②③。
3. 跑 **`--no-audio` ablation**，量 audio 的實際貢獻。
4. 若 PASS（Qwen 夠接近）→ 用 Qwen 標滿 2000hr；**只把 Qwen 低信心（低 confidence）或 overlap 的 turn
   丟回 Gemini/人工複核**（active-learning，省錢又保品質）。
5. 把 turn-level addressee 展開成 frame-level `<ads>` token stream，接進訓練 data-prep（TODO 2）。
```
