# ttv2（turn_taking_2000_v2）資料生成錯誤清單

檢查 234 段對話、4,218 句的 `script.jsonl`（文字與 `addressing` 標籤），並用 Parakeet ASR 抽查音檔實際念了什麼。
完整逐句清單：[`data/ttv2/script_issues.csv`](../data/ttv2/script_issues.csv)（每列一句一種問題，含 `canonical_text`、
下一位說話者、建議修法）；統計：`data/ttv2/script_issues_summary.json`。重跑：

```bash
python3 scripts/15_audit_ttv2_script.py --root /path/to/turn_taking_2000_v2 \
    --out-csv data/ttv2/script_issues.csv --out-json data/ttv2/script_issues_summary.json
```

（2026-10-12，程式檢查，`human_audited=false`。偵測是規則式的，可能有漏網或誤判。）

## 總覽

| # | 問題 | 句數 | 對話數 | 音檔受影響？ | 建議修法 |
|---|---|---|---|---|---|
| 1 | 自己叫自己的名字 | 51 | 38 | 是（TTS 念出自己的名字） | 修生成器，重生這些句子 |
| 2 | 開場「guys」句被標成對下一位說話者 | 65 | 65 | 否（只有標籤錯） | 改標籤為 EVERYONE |
| 3 | 座位字母當名字 `[C]` | 26 | 14 | 是（名字從沒念出來） | 換成名字後重生 |
| 4 | 舞台指示 `[to C]` 寫進台詞 | 2 | 1 | 是（念成 "To see"、"To me"） | 從台詞拿掉 |
| 5 | 其他括號詞 `[EVERYONE]` `[sighs]` `[BRAN]` | 5 | 5 | `[sighs]` 念成 "size"；`[BRAN]` 被吞掉；`[EVERYONE]` 正常念 "Everyone" | 拿掉或換成正確的字 |
| 6 | 座位字母當名字（無括號）"A and B tackle…" | 1 | 1 | 是（念出字母） | 換成名字後重生 |
| 7 | 名字留著括號 `[Jerome]` `[MABEL]` | 144 | 47 | 否（TTS 正常念名字） | 只要清掉文字裡的括號 |

1–6 合計影響 95 段對話（234 段的 41%）。

## 1. 自己叫自己的名字（51 句 / 38 段）

說話者一開口就叫自己的名字，下一句又是同一人回答自己的問題：

```
tr5191  Emil:  Emil, did you manage to pay your share?     (addressing: EVERYONE)
        Emil:  Yes, I put down a deposit earlier today.
```

- **51 句全部**都是「下一句由同一人回答」，`addressing` 全部是 EVERYONE。
- `script_canonical.jsonl` 裡就已經錯了（canonical 是 "Casey, would you mind…" 由 Casey 說），所以不是換名字（name remap）
  造成的，是生成階段：被點名的人和提問的人被分配到同一個座位。
- 音檔真的念了自己的名字（dev20 ASR 確認："Marilyn, would you mind…"、"ML, did you manage…"）。
- 建議：提問句應由另一個座位說、`addressing` 設成被點名的人；修好後重生這些句子的音檔。
  標註端（2026-10-12 起）：`13_build_frames_v5.py` 把這些句子在其他座位的 view 中遮罩（說話者自己仍是 `self`）。

## 2. 開場 guys 句被標成對單一個人（65 句 / 65 段）

```
tr0530  Glen:  Alright guys, we need to figure out our plan for tomorrow's camping trip.   (addressing: B)
```

- 65 句裡 64 句是**對話的第一句**，而且標的都是**下一位說話者**；文字明明是對大家說，也沒提到那個人的名字。
- 看起來是 `repair_dialogue()` 為了湊足「指定對象」的句數，把第一句改標成下一位說話者。
- 只影響 `addressing` 標籤，音檔沒問題。建議改回 EVERYONE。
  （我們的 ADS v3 標註不用 generator 標籤，所以不受影響；但用 `addressing` 訓練或評估的人會受影響。）

## 3. 座位字母當名字 `[C]`（26 句 / 14 段）

```
tr2206  Roy:  [C], do you think you could get us there?     (addressing: C, C = Leona)
```

- canonical 腳本裡就是 `[C]`，生成器輸出座位字母而不是名字，之後也沒有被換成名字。24/26 句 `addressing` 和字母一致。
- 音檔：ASR 檢查 26 句：約 15 句字母被吞掉（只剩 "Do you think…"），約 10 句念出字母本身（"D, will you…"、"Dee"、"B. Did you…"）。**兩種情況名字都沒有出現在音檔裡**，聽的人無從知道是對誰說。
- 影響：LLM 標註看到 `[C]` 不知道是誰，dev20 的 5 句錯了 2 句（猜成上一位說話者）。
- 建議：換成該座位的名字後重生音檔。
- 標註端（2026-10-12 起）：`00c_prepare_ttv2.py` 預設把 `[C],` 這種呼喚拿掉、其他位置換成 "someone"，
  讓 LLM 看到的和音檔聽得到的一致（`--keep-seat-letters` 可關掉，EXP-062 dev20 用的是舊行為）。

## 4. 舞台指示寫進台詞（2 句，tr2188）

```
tr2188  Austin:  [to C] Do you think we can manage the time sensitivity?
tr2188  Daphne:  [to me] Can you help drive around while we work on it?
```

音檔：TTS 把它念出來了（ASR："To see, do you think…"、"To me, can you help…"）。建議從台詞拿掉，改放在 metadata。我們這邊這兩句已遮罩。

## 5. 其他括號詞（5 句）

| 對話 | 句子 | 問題 |
|---|---|---|
| tr2120 | `[EVERYONE], does anyone know any good spots nearby?` | 標籤值寫進台詞 |
| tr2196 | `[EVERYONE], I think we should try folding…` | 同上 |
| tr2158 | `[sighs] Okay, but just the main points please!` | 非語言標記寫進台詞 |
| tr2189 | `[sighs] I guess I'll work on those…` | 同上 |
| tr2245 | `[BRAN], what tasks can you manage today…`（addressing B = Daryl） | 名字壞掉（不是任何參與者，疑似 canonical "Brian" 殘留） |

音檔：（ASR）`[EVERYONE]` 念成 "Everyone"，可以接受，只需要清掉括號；`[sighs]` 被念成 "size"，要重生；`[BRAN]` 被吞掉（"What tasks can you…"），名字沒出現。

## 6. 無括號的座位字母（1 句，tr5170）

`How about A and B tackle the big pieces while D and I handle smaller bits?`：TTS 會照念 "A and B"。

## 7. 名字留著括號（144 句 / 47 段，低優先）

`[Jerome], can you tell us…`、`[MABEL], can you check…`：TTS 會正常念出名字（ASR 確認），只是文字不乾淨，
做文字處理時要記得去掉括號。大寫的 `[MABEL]`、`[DEBRA]`、`[JOSEPHINE]` 等也屬此類。

## 其他觀察（不是錯，但要知道）

- `script.jsonl` 的 `start/end` 和音檔實際時間最多差約 1.5 秒；script 的時間窗彼此不重疊，但音檔裡有重疊。
  需要時間的地方請以音檔為準（我們用 VAD + ASR）。
- generator 的 `addressing` 有 333/855 個指定對象的標籤，文字裡沒有該人名字（多數是「下一位說話者」），
  這是標註政策問題，不一定是錯。
