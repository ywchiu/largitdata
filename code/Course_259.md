# Claude Code 多模型示範流程（步驟版）

Opus 規劃 + Codex 審查 + GLM / DeepSeek 執行

---

# 1. 安裝

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @musistudio/claude-code-router
npm install -g @openai/codex
```

# 2. 環境變數

```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
```

寫入 `~/.zshrc` 後 `source ~/.zshrc`。
（`NODE_EXTRA_CA_CERTS` 必填，否則 CCR 會 `fetch failed: unable to get local issuer certificate`。）

# 3. CCR 設定

```bash
mkdir -p ~/.claude-code-router
nano ~/.claude-code-router/config.json
```

```json
{
  "LOG": true,
  "API_TIMEOUT_MS": 900000,
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "$OPENROUTER_API_KEY",
      "models": [
        "anthropic/claude-opus-4.8",
        "z-ai/glm-5.2",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1"
      ],
      "transformer": {
        "use": [
          "openrouter",
          ["maxtoken", { "max_tokens": 65536 }],
          "enhancetool",
          "reasoning"
        ]
      }
    }
  ],
  "Router": {
    "think": "openrouter,anthropic/claude-opus-4.8",
    "default": "openrouter,z-ai/glm-5.2",
    "background": "openrouter,deepseek/deepseek-chat",
    "longContext": "openrouter,z-ai/glm-5.2",
    "longContextThreshold": 120000
  }
}
```

確認 opus slug 是否存在：

```bash
curl -s https://openrouter.ai/api/v1/models | grep -o '"id":"anthropic/claude-opus[^"]*"'
```

# 4. 啟動

```bash
ccr restart
ccr model      # 查看路由
ccr code       # 開多模型 Claude Code
```

# 5. 驗證路由（用 /v1/messages，不是 /v1/chat/completions）

```bash
# default → GLM-5.2
curl -s http://127.0.0.1:3456/v1/messages \
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":512,"messages":[{"role":"user","content":"Reply with exactly ROUTE_OK"}]}'

# think → Opus-4.8（帶 thinking 才觸發）
curl -s http://127.0.0.1:3456/v1/messages \
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":2048,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"Reply with exactly ROUTE_OK"}]}'

# background → DeepSeek
curl -s http://127.0.0.1:3456/v1/messages \
  -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-3-5-haiku","max_tokens":512,"messages":[{"role":"user","content":"Reply with exactly ROUTE_OK"}]}'
```

回應 JSON 的 `"model"` 欄位即為實際服務的模型。

# 6. 安裝 codex-plugin-cc

Claude Code 內：

```text
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

確認 codex CLI 已登入：

```bash
codex login status
codex exec --skip-git-repo-check "Reply with exactly CODEX_OK"
```

# 7. Repo 檔案結構

```text
docs/agent-plans/
  <task>.md              # 原始 plan
  <task>.codex-review.md # Codex 審查
  <task>.final.md        # 最終 plan（executor 唯一依據）
```

---

# 工作流步驟

## Step 1：Opus 產生 plan

```text
請只做 planning，不要修改 source code。

目標：實作 <任務>。

請先讀 codebase，產出 implementation plan 到 docs/agent-plans/<task>.md，內容包含：
1. Goal
2. Current system understanding
3. Affected files
4. Implementation phases
5. Test strategy
6. Regression risks
7. Rollback plan
8. Stop conditions

限制：不要改 code、不要開始實作、不要擴張 scope，完成後停下等批准。
```

## Step 2：Codex adversarial review

```text
/codex:adversarial-review --background
請只審查 docs/agent-plans/<task>.md，不要實作。以 adversarial architecture reviewer 角度挑戰：
1. 是否解決真正問題  2. 是否找對檔案/入口  3. hidden coupling  4. phase 是否太大
5. 測試是否足夠  6. rollback 是否具體  7. 是否破壞既有行為  8. 必須 block 的事
9. 更簡單更安全的做法

輸出：Blockers / Major / Minor / Suggested edits / Final recommendation。
```

```text
/codex:status
/codex:result
```

結果存成 `docs/agent-plans/<task>.codex-review.md`。

## Step 3：Opus 整合 review

```text
請讀取 docs/agent-plans/<task>.md 與 <task>.codex-review.md，
產出 docs/agent-plans/<task>.final.md。

規則：不要改 code；對每個 Codex blocker/major 標記 accept/reject/partial + reason；
更新 phases、test strategy、rollback、stop conditions；final plan 為唯一 authoritative source。
```

## Step 4：Human 批准

```text
我批准 docs/agent-plans/<task>.final.md。請進入 implementation：
1. 僅依 final plan  2. 一次一個 phase  3. 每 phase 完成後停下回報
4. 不擴張 scope  5. 不自行改架構  6. 跑 final plan 指定 tests
7. 發現 final plan 錯誤立刻停止。
```

## Step 5：GLM / DeepSeek 執行 Phase

```text
請執行 docs/agent-plans/<task>.final.md 的 Phase 1。
只做 Phase 1、不擴張 scope，完成後停下並回報：
修改檔案 / 改了什麼 / 跑了哪些測試 / 測試結果 / 是否偏離 plan / 下一步建議。
```

## Step 6：每個 phase 驗收

```text
請輸出 Phase <N> completion report：
Files changed / Summary / Tests run / Results / Failures / Deviations / Risk / Next-phase recommendation。
```

測試失敗時：

```text
測試失敗，請不要繼續下一個 phase。先分析：
失敗原因 / 是 implementation 錯還是 plan 錯 / 最小修正方案 / 是否需回 Opus 重修 plan。
```

## Step 7：跑完整測試

```text
請依 final plan 執行完整測試：Unit / Integration / Regression / Golden / Manual smoke / 已 skip 的測試與原因。
```

## Step 8：Codex review final diff

```text
/codex:review --base main --background
```

高風險改動：

```text
/codex:adversarial-review --base main --background
請審查這次 diff，關注：是否偏離 final plan / hidden regression / 安全 / 測試不足 / 過度設計 / production risk / 更小 patch。
輸出：Blockers / Major / Minor / Required fixes / Final recommendation。
```

```text
/codex:status
/codex:result
```

## Step 9（選）：Opus final review

```text
請只做 final review，不要改 code。讀取 final plan / current diff / test results / Codex final review，判斷：
是否符合 final plan / 是否有未處理 blocker / 測試是否足夠 / 是否可 merge / 否則列最小修正清單。
輸出：Approve / Approve with minor fixes / Reject。
```

## Step 10：Human merge gate

```text
[ ] final plan exists
[ ] Codex plan review exists
[ ] Codex concerns reconciled
[ ] implementation follows final plan
[ ] tests passed
[ ] final diff reviewed
[ ] human approved
```

---

# 常見錯誤

| 症狀 | 解法 |
| --- | --- |
| `Provider 'undefined' not found` | 改用 `/v1/messages`（非 `/v1/chat/completions`） |
| `fetch failed: unable to get local issuer certificate` | `export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem` 後重啟 |
| opus `Provider not found` | slug 失效，改用實際存在的固定 slug |
| 回應 `content` 為空 | reasoning model，調高 `max_tokens`（≥512） |

---

# 範例：在 vibe-backtester 新增 RSI 策略

專案：`/Users/david/course/vibe-backtester`（FastAPI 回測系統）
任務 slug：`rsi-strategy`
目標：仿照既有 MA 策略（`indicators/ma_indicator.py`、`backtest/ma_backtest.py`、`api/ma_routes.py`、`tests/test_ma_strategy.py`），新增一條 RSI 策略。屬 Level 3，跑完整流程。

## 0. 在專案目錄啟動

```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
export NODE_EXTRA_CA_CERTS=/etc/ssl/cert.pem
ccr restart
cd /Users/david/course/vibe-backtester
mkdir -p docs/agent-plans
ccr code
```

## 1. Opus 產生 plan

```text
請只做 planning，不要修改 source code。

目標：在本專案新增一條 RSI 策略，行為對照既有的 MA 策略。
請參考 backend/indicators/ma_indicator.py、backend/backtest/ma_backtest.py、
backend/api/ma_routes.py、backend/api/models.py、backend/tests/test_ma_strategy.py
的既有模式。

請先讀 codebase，產出 implementation plan 到 docs/agent-plans/rsi-strategy.md，內容包含：
1. Goal  2. Current system understanding  3. Affected files  4. Implementation phases
5. Test strategy  6. Regression risks  7. Rollback plan  8. Stop conditions

限制：不要改 code、不要開始實作、不要擴張 scope，完成後停下等批准。
```

## 2. Codex adversarial review

```text
/codex:adversarial-review --background
請只審查 docs/agent-plans/rsi-strategy.md，不要實作。以 adversarial architecture reviewer 角度挑戰：
是否找對檔案/入口、是否漏掉與 ma 共用的 service/route 註冊、RSI 計算邊界條件、
phase 是否太大、測試是否足夠、rollback 是否具體、是否破壞既有 MA 行為、更簡單的做法。
輸出：Blockers / Major / Minor / Suggested edits / Final recommendation。
```

```text
/codex:status
/codex:result
```

把結果存成 `docs/agent-plans/rsi-strategy.codex-review.md`。

## 3. Opus 整合 review

```text
請讀取 docs/agent-plans/rsi-strategy.md 與 docs/agent-plans/rsi-strategy.codex-review.md，
產出 docs/agent-plans/rsi-strategy.final.md。
對每個 Codex blocker/major 標記 accept/reject/partial + reason；
更新 phases、test strategy、rollback、stop conditions；不要改 code。
```

## 4. Human 批准

```text
我批准 docs/agent-plans/rsi-strategy.final.md。請進入 implementation：
僅依 final plan、一次一個 phase、每 phase 完成後停下回報、不擴張 scope、
不自行改架構、跑 final plan 指定 tests、發現 plan 錯誤立刻停止。
```

## 5. GLM / DeepSeek 執行 Phase 1

```text
請執行 docs/agent-plans/rsi-strategy.final.md 的 Phase 1。
只做 Phase 1、不擴張 scope，完成後停下並回報：
修改檔案 / 改了什麼 / 跑了哪些測試 / 測試結果 / 是否偏離 plan / 下一步建議。
```

## 6. 跑測試

```text
請依 final plan 執行測試：
python -m pytest backend/tests -q
回報 Files changed / Tests run / Results / Failures / Deviations。
```

## 7. Codex review final diff

```text
/codex:review --base main --background
```

```text
/codex:status
/codex:result
```

## 8. ChatGPT 最終 review

開發完成後，用 ChatGPT（codex 後端即 ChatGPT 登入）對完整 diff 做最後一次 review。

在 Claude Code 內：

```text
/codex:adversarial-review --base main --background
請以資深 reviewer 角度做最終 review：正確性、是否破壞既有 MA 行為、
測試是否足夠、安全與 production 風險、是否有更小的 patch。
輸出：Blockers / Required fixes / Final recommendation（Approve / Approve with fixes / Reject）。
```

或在終端機直接用 codex CLI 跑：

```bash
git -C /Users/david/course/vibe-backtester diff main > /tmp/rsi-diff.txt
codex exec --skip-git-repo-check "你是資深 code reviewer，請 review 以下 diff：\
正確性、是否破壞既有 MA 策略行為、測試是否足夠、安全與 production 風險，\
最後給出 Approve / 需修正清單。\n\n$(cat /tmp/rsi-diff.txt)"
```

## 9. 成本計算（不同模型組合比較）

review 完之後，計算這次任務若用不同模型組合各要多少錢。

各模型定價（每 1M tokens，OpenRouter / Anthropic 官方，2026-06）：

| 模型 | input | output |
| --- | --- | --- |
| Opus 4.8 | $5.00 | $25.00 |
| Sonnet 4.6 | $3.00 | $15.00 |
| GLM-5.2 | $1.20 | $4.10 |
| DeepSeek-chat | $0.20 | $0.80 |
| DeepSeek-R1 | $0.70 | $2.50 |
| gpt-5.3-codex (Codex review) | $1.75 | $14.00 |

用 `cost_compare.py`（把各階段 token 換成 CCR log 的實際值）：

```bash
python cost_compare.py
```

範例輸出（以 RSI 任務粗估 token，全部用 API 費用計入）：

```text
scenario            total    vs all-opus
------------------------------------------
all-opus       $    2.950     (baseline)
opus+sonnet    $    1.908           -35%
multi-model    $    1.196           -59%
```

組合定義（在 `cost_compare.py` 的 `SCENARIOS`，review 固定走 Codex）：

```text
all-opus     所有階段都用 Opus 4.8（含 review）
opus+sonnet  plan/reconcile 用 Opus，execute/background 用 Sonnet，review 用 gpt-5.3-codex
multi-model  plan/reconcile 用 Opus，execute 用 GLM-5.2，background 用 DeepSeek，review 用 gpt-5.3-codex
```

> 全部以 API 單價計入，含 Codex review（gpt-5.3-codex）。若 Codex 走 ChatGPT 訂閱則該段為固定月費、非 per-token。
> token 數要從 CCR log（`~/.claude-code-router/logs/`，`model_usage` 欄位）或各模型
> 回應的 usage 換成實際值，金額才準。

## 10. Human merge gate

```text
[ ] docs/agent-plans/rsi-strategy.final.md exists
[ ] Codex plan review exists
[ ] implementation follows final plan
[ ] pytest backend/tests 通過
[ ] final diff reviewed
[ ] ChatGPT 最終 review 通過（無 blocker）
[ ] 成本計算完成（多模型 vs all-opus）
[ ] human approved
```
