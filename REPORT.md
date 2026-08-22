# Addressee Labeling 進度報告

> Multi-Speaker Full Duplex 專案 · 投影片 p.18 TODO 1
> 「Label current datasets the new Addressee information」
> 更新於 2026-08-22

---

## 1. 目標

為多人對話語料標註 **addressee**(`<ads>`):每句話是講給誰聽的。
這是訓練階段二(讓模型生成 `<ads>` token)的前置資料。

**核心問題**:能否用便宜、可自架的 Qwen 取代昂貴的商用模型來標註全部語料?

---

## 2. 方法

```
Golden (強模型)  ─┐
                  ├─→ 逐句比對 → 一致性夠高?→ 用 Qwen 標全部語料
Candidate (Qwen) ─┘
```

- **Label 集合**(封閉):其他 speaker id / `GROUP`(對全場) / `UNKNOWN`(無法判斷),可多選
- **控制變因**:golden 與 candidate 使用**完全相同的 prompt**,只差模型
- **評估指標**:exact match、Jaccard、Cohen's kappa、per-class F1,並依 overlap / backchannel / 人數分層
- **通過門檻**(事前訂定):kappa ≥ 0.75 且 exact ≥ 0.80

**資料**:AMI 會議語料 ES2002a(4 speakers、236 句、18.5 分鐘、75% 句子有重疊)

---

## 3. 準確度結果

### 3.1 Golden 交叉驗證(新增)

與另一組獨立標註結果比對(Gemini,**audio+transcript**,涵蓋 9 場會議 30 個 3 分鐘 clip)。
重疊範圍為 ES2002a 的 3 個 clip,共 **107 句**。

| 配對 | kappa | exact |
|---|---|---|
| Gemini Flash vs Gemini Pro(同家族) | 0.948 | 0.945 |
| **GPT-5 vs Gemini Pro** | **0.894** | 0.916 |
| **GPT-5 vs Gemini Flash** | **0.895** | 0.916 |
| Gemini Flash vs Qwen3-32B(調校 prompt) | 0.845 | 0.881 |
| Gemini Pro vs Qwen3-32B(調校 prompt) | 0.829 | 0.853 |

**意義**:

1. **Golden 可信度大幅提升**——兩家不同廠商的旗艦模型獨立標註,一致性達 0.89,不再是「單一模型說了算」。
2. **音檔的增益可能有限**——GPT-5 為純文字、Gemini 為 audio+transcript,兩者仍達 0.89。
3. **但無法斷定**——同家族的 Flash vs Pro 為 0.948,高於跨家族的 0.894。這 0.05 落差可能來自模型家族差異而非音檔,**兩因素在現有資料中無法分離**。

### 3.2 Prompt 的影響大於模型大小(新增)

| 設定 | kappa |
|---|---|
| Qwen3-14B + 原始 prompt(本專案) | 0.616 |
| Qwen3-32B + 調校 prompt(`v2_adjacency_backchannel`) | **0.845** |

外部團隊的 prompt bake-off 顯示:同為 Qwen3-32B,原始 prompt 為 0.645、最佳版本可達 0.717(對 Flash golden)。
**Prompt 工程的效益顯著,應優先於加大模型。**

### 3.3 模型大小(Thinking ON,30 句子集,原始 prompt)

| 模型 | exact | kappa | macro F1 |
|---|---|---|---|
| Qwen3-4B | 0.667 | 0.587 | 0.695 |
| Qwen3-14B | **0.700** | **0.616** | 0.515 |

主要弱點(`GROUP` 被誤判為 `B`)隨模型變大改善:誤判 7 次 → 3 次,GROUP recall 0.429 → 0.643。

### 3.4 Thinking Mode 不可關閉

| 模型 | Thinking ON | Thinking OFF | 變化 |
|---|---|---|---|
| Qwen3-4B | 0.587 | 0.284 | −0.303 |
| Qwen3-14B | 0.616 | 0.178 | −0.438 |

原因:關閉後模型大量輸出 `UNKNOWN`(放棄判斷)。golden 僅 1 句,4B 為 72 句(31%),14B 更達 146 句(62%)——模型越大越嚴格遵守 prompt 中「不確定就選 UNKNOWN」的指示,分數反而更低。

> 註:兩組使用不同資料量與 window 設定,數值不宜直接相減;但 UNKNOWN 比例差異足以說明趨勢。

---

## 4. 速度結果

同一份 4.1 分鐘逐字稿,H100 NVL 單卡,所有模型設定相同。

| 模型 | HF (tok/s) | vLLM (tok/s) | 加速 |
|---|---|---|---|
| Qwen3-4B | 17.4 | 282.9 | 16.3× |
| Qwen3-8B | 失敗 | 188.3 | — |
| Qwen3-14B | 16.5 | 114.1 | 6.9× |
| Qwen3-32B | 12.8 | 52.2 | 4.1× |

**關鍵發現**:HF 下 4B 與 14B 僅差 5%(參數量差 3.5 倍卻幾乎同速),代表測到的是 Python 逐 token 開銷而非模型速度。vLLM 下 4B→32B 拉開 5.4 倍,才符合記憶體頻寬受限的理論預期。

### 200 小時語料處理時間推估(vLLM)

| 模型 | 所需時間 |
|---|---|
| Qwen3-4B | 9.2 小時 |
| Qwen3-14B | 21.0 小時 |
| Qwen3-32B | **42.7 小時(1.8 天)** |

> 僅計生成時間,不含一次性模型載入;循序處理、未使用 vLLM 連續批次,故為保守上限。

---

## 5. 結論

1. **Golden 已通過跨廠商交叉驗證**(GPT-5 vs Gemini = 0.894),可信度足以作為評估基準。
2. **Prompt 比模型大小更關鍵**——調校 prompt 的 32B 達 0.845,遠高於原始 prompt 的 14B(0.616)。
3. **速度不是瓶頸**——即使 32B,200 小時也僅需 1.8 天,應純粹以準確度作為選型依據。
4. **必須使用 vLLM**,HF 僅適合開發除錯;**Thinking mode 必須開啟**。
5. **本專案自身的 Qwen 尚未達標**(最佳 0.616 < 0.75),但外部結果顯示 **0.845 可達成**,路徑明確。

---

## 6. 下一步

| 優先 | 項目 |
|---|---|
| **高** | 導入調校後的 prompt(`v2_adjacency_backchannel`),以 vLLM 重跑 32B——預期可從 0.616 提升至 0.8+ |
| **高** | 擴大測試樣本(目前僅 ES2002a 單場),沿用外部的 9 場 30 clip 設計 |
| 中 | 建立**人工標註 micro-gold**(50–200 句),量測 golden 本身的準確率 |
| 中 | Audio ablation:讓同一模型分別跑有/無音檔,以分離「音檔」與「模型家族」兩個因素 |
| 低 | 針對 `GROUP` / `UNKNOWN` 的判準持續改善 |

---

## 7. 現況與限制

- 本專案 golden 為 **GPT-5(純文字)**;原設計的 Gemini + 音檔因 Pro 級模型免費額度為 0 而未執行,該部分由外部結果補足。
- 交叉驗證僅涵蓋 **107 句**(外部 1136 句中的 9%),因其餘 8 場會議本專案尚未標註。
- 所有數字皆為**標註者之間的一致性,非準確率**——兩側均無人工驗證的 ground truth。
- 稀有類別(`C` 僅 1 句、`UNKNOWN` 為 0)會嚴重拖低 macro F1,判讀時應以 **kappa** 為準。

---

## 附錄:重現方式

```bash
# 資料
bash scripts/download_ami.sh ES2002a

# ① golden  ② candidate  ③ 評估  ④ 合併回逐字稿
python3 scripts/01_build_golden.py --backend openai --model gpt-5 --no-audio \
    --conversations data/ami/es2002a_full.jsonl --out outputs/golden.jsonl
python3 scripts/02_run_candidate.py --backend local --engine vllm \
    --model Qwen/Qwen3-32B --conversations data/ami/es2002a_full.jsonl \
    --out outputs/candidate.jsonl
python3 scripts/03_evaluate.py --conversations data/ami/es2002a_full.jsonl \
    --gold outputs/golden.jsonl --pred outputs/candidate.jsonl
python3 scripts/04_merge_labels.py --conversations data/ami/es2002a_full.jsonl \
    --labels outputs/candidate.jsonl --out-jsonl outputs/labeled.jsonl

# 速度 benchmark / 與外部標註比對
python3 scripts/05_benchmark_speed.py --conversations data/ami/bench_5min.jsonl \
    --models Qwen/Qwen3-4B Qwen/Qwen3-32B --engine vllm
python3 scripts/06_compare_external.py --external-dir <bundle> \
    --external-labels labels/gemini_3.1_pro.jsonl \
    --my-conversations data/ami/conversations.jsonl \
    --my-labels outputs/golden.jsonl
```

Repo: https://github.com/Justintsai502/addressee-labeling
