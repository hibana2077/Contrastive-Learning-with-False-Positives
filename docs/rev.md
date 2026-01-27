# Reviewer Comments on Main Results (Section 3)

## 整體評價（Summary）

本章系統性地建立了一個「time–space decoupled」長影片壓縮的可證明理論基線，三個主要定理（Theorems 1–3）在邏輯上層層遞進，從**可行性（screening）**、到**端到端效能（estimation）**、再到**不可避免的資訊理論限制（lower bound）**，結構完整且內在一致。結果在假設清楚、依賴關係透明的前提下，給出了明確的 token 複雜度與誤差分解，對理論社群具有相當價值。

---

### 1. Theorem 1（Temporal Screening Guarantee）

**優點：**

* 定理清楚刻畫了在 **margin κ = ∆ − 2γ > 0** 下，僅需沿時間軸進行低成本抽樣，即可高機率偵測所有事件邊界，並將誤檢數量控制在 (O(s + \log T))。
* 使用投影後的一維統計量 (g_t = |\hat a_t - \hat a_{t-1}|)，成功避免維度依賴，與後續 remark 中的「task-aligned screening」論述相互呼應。
* 對 ±1 localization error 的討論誠實且合理，並明確說明其對 Stage II 幾乎無實質影響。

**需要澄清或可加強之處：**

* 定理高度依賴 separability 假設 ∆ > 2γ。雖然作者引入 κ 並解釋其角色，但目前的敘述仍偏「硬門檻（hard threshold）」。建議在主文中更直觀地說明：當 κ 接近 0 時，失敗是如何具體發生的（例如 false positive 爆炸或 miss detection 的相變行為）。
* τ 的「admissible interval」雖在附錄中提供資料驅動校準，但在主結果中仍顯得略為理想化。可考慮在定理後直接給出一個簡化、但實務可用的 τ 設定方式（即使常數較鬆）。

---

### 2. Theorem 2（End-to-End Score Accuracy）

**優點：**

* 誤差分解為 **統計誤差 ε** 與 **結構性漂移誤差 O(γL)** 非常清楚，這是本章最具洞見的部分之一。
* 漂移誤差項並非以隱含方式吸收，而是明確呈現為 deterministic bias，並在後文證明其 tightness，展現理論上的誠實性。
* Token budget (式 (10)) 將 rT、anchor overhead、與 refinement cost 清楚分離，使 time–space decoupling 的節省來源一目了然。

**需要澄清或可加強之處：**

* L 的選擇對整體表現極為關鍵，但目前最佳化 L 的討論僅以「typical operating regime」形式出現。建議明確指出在給定 (ε, γ, T) 下，如何系統性地選擇 L（即使僅是 order-wise）。
* 當 γ 並非顯著小於 ε 時（例如中度或快速漂移），該方法是否仍具優勢？目前結果在形式上仍成立，但實務意義可能下降，建議在文字中更直接說明適用範圍。

---

### 3. Theorem 3（Information-Theoretic Lower Bound）

**優點：**

* Lower bound 與前述結果形成良好對照，明確指出目前 Stage I 掃描成本 rT 與最優 Ω(s log(T/s)) 之間仍存在 gap。
* 定理對 fully adaptive querying 仍成立，且 dimension-robust，成功避免被質疑為「模型過弱」。
* 將漂移模型視為常數背景的超集合，邏輯嚴謹，論證乾淨。

**需要澄清或可加強之處：**

* Lower bound 僅適用於「定位事件時間」，而非直接針對最終 score estimation。雖然這在文中是正確且合理的，但建議更明確提醒讀者：此下界不直接否定存在「不定位但仍能估分」的策略。
* 若能在討論中簡要指出，哪些額外結構（例如 multiscale 或 group testing）可能有助於逼近該下界，將有助於未來研究定位。

---

### 4. 結構與表述層面的整體建議

* Table 1 對讀者極為友善，但「Key conditions」與「token scaling」之間的對應關係仍略顯密集。可考慮在 caption 或正文中加入一句解讀指引。
* 本章在技術上相當成熟，但對非該子領域讀者而言，γ、∆、κ、L 同時出現仍具負擔。適度加入一個「parameter roles」的小結，將有助於可讀性。

---

### 總結（Overall Recommendation）

Main Results 章節在理論深度、結構完整性與誠實性方面表現優異，是一組**乾淨、可解釋、且具參考價值的基線結果**。若能在 separability 假設的實際影響、L 的選擇策略、以及 lower bound 與 estimation 任務的關係上補充更直觀的說明，將使本章不僅「正確」，也更「好用」。
