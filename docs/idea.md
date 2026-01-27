# Provable Time–Space Decoupling for Long-Form Video Compression

---

## 1. 問題設定與符號

* 影片長度：(T) 個時間步（可視為幀或短片段）。
* 每個時間步有 (N) 個空間 tokens：({x_{t,i}\in\mathbb{R}^d}_{i=1}^N)。
* 我們允許演算法「讀取」部分 tokens，總讀取數（token budget）記為 (B)。

### 下游任務（刻意選 COLT 友好的族）

先用最容易分析、但仍能對應 HOI/NLQ 的任務族：

**(A) 線性打分/分類任務：**
存在未知 (w\in\mathbb{R}^d)，(|w|*2\le 1)，真實打分
[
s^\star ;=; \frac{1}{T}\sum*{t=1}^T \Big\langle w,;\mu_t \Big\rangle,
\quad
\mu_t := \frac{1}{N}\sum_{i=1}^N x_{t,i}.
]
預測為 (\hat y = \mathrm{sign}(s)) 或用 hinge/logistic loss 分析。

這足夠涵蓋「是否出現某個事件/動作概念」這種 long video 判別，也能作為更複雜 head 的理論替身（你可在實作中換成多類別或對比式檢索，但理論先站穩）。

---

## 2. 資料生成模型（稀疏事件 + 穩定背景）

### 2.1 事件集合

存在未知事件時間集合 (E\subseteq{1,\dots,T})，其結構為 (s) 個不交疊事件區間的聯集：
[
E = \bigcup_{j=1}^s [a_j, b_j],\qquad 1\le a_j\le b_j\le T.
]
你可加一個最常用且好證明的分離假設（事件之間至少隔 (\Delta_{\text{sep}})）。

### 2.2 背景穩定性（避免 joint 時空難題的關鍵）

對每個時間 (t)，token 由「背景 + 事件增量 + 噪聲」組成：
[
x_{t,i} = m_t + \mathbf{1}{t\in E},\delta_t + \xi_{t,i}.
]

* 背景均值 (m_t) 緩慢漂移：(|m_{t}-m_{t-1}|_2 \le \gamma)（小漂移）。
* 事件增量 (\delta_t) 在事件期間有最小強度：(|\delta_t|_2 \ge \Delta)。
* 噪聲 (\xi_{t,i}) 為零均值、(\sigma^2)-subGaussian（或 bounded）：
  (\mathbb{E}[\xi_{t,i}]=0)，且有標準集中不等式可用。

直覺：長時間背景相對穩定（(\gamma) 小），事件期間特徵均值發生顯著偏移（(\Delta) 大）。這正是 Ego4D 長影片常見型態。

---

## 3. 壓縮目標（你要證明什麼）

演算法讀取 (B) 個 tokens 後，輸出壓縮估計 (\hat s) 近似 (s^\star)，並使得下游損失退化可控。

最乾淨的目標是 **打分誤差**：
[
|\hat s - s^\star| \le \varepsilon
]
或 **分類/損失差**（對 1-Lipschitz 損失 (\ell)）：
[
\ell(\hat s, y)-\ell(s^\star,y)\le \varepsilon.
]

---

## 4. 兩階段演算法（Time → Space），完全對應 STIM-TM 的直覺但可證明

### Step 1：時間偵測（用少量 token/幀摘要找事件區間）

對每個時間 (t)，抽樣 (r) 個 tokens（(r\ll N)）估計該幀均值：
[
\hat\mu_t = \frac{1}{r}\sum_{j=1}^r x_{t, I_{t,j}},
\quad I_{t,j}\sim \text{Uniform}({1,\dots,N}).
]
計算變化分數（change score）：
[
g_t = |\hat\mu_t - \hat\mu_{t-1}|_2.
]
以閾值 (\tau) 偵測事件邊界，得到候選事件時間集合 (\widehat{E})（或候選區間集合 (\widehat{\mathcal{I}})）。接著對每個候選區間選代表時間點（例如取區間中點或最大分數點）形成集合 (\widehat{T})，其大小 (k:=|\widehat{T}|) 會與 (s) 同階。

> 這一步的理論核心：subGaussian 均值估計 + 變化檢定的偵測/誤警機率控制。

### Step 2：空間壓縮（只在 (\widehat{T}) 上花 token）

對每個被選時間 (t\in\widehat{T})，再抽樣 (m) 個 tokens（(m\ll N)）估計更精準的 (\tilde\mu_t)，用以近似真均值 (\mu_t)。最後用
[
\hat s ;=; \frac{1}{T}\sum_{t\notin \widehat{T}} \langle w, \tilde m_t\rangle
;+; \frac{1}{T}\sum_{t\in \widehat{T}} \langle w, \tilde\mu_t\rangle
]
其中 (\tilde m_t) 可用「背景段落的 pooled 均值」或「鄰近非事件代表」替代（最簡單做法：用分段後每段一個背景代表均值）。

> 這一步的理論核心：均值估計誤差如何轉成線性打分誤差。

---

## 5. 主定理（可直接寫在 paper 的 Theorem 1/2/3）

### Theorem 1（事件偵測：高機率召回 + 有界誤警）

在上述模型下，取
[
r ;\ge; C_1\frac{\sigma^2}{(\Delta-2\gamma)^2}\log\frac{T}{\eta},
\qquad
\tau = \frac{\Delta}{2}
]
且假設 (\Delta>2\gamma)。則以機率至少 (1-\eta)：

1. 所有真正事件邊界都被偵測（或所有事件區間都與 (\widehat{\mathcal{I}}) 有交集）。
2. 誤警數（偽邊界）至多 (O(\log(T/\eta)))（或更強：期望 (O(1)) 且高機率小於常數倍）。

因此可選出 (k=O(s+\log(T/\eta))) 個代表時間點。

**證明骨架：**

* 用 subGaussian 集中不等式界 (|\hat\mu_t-\mu_t|_2) 的尾機率（union bound over (t)）。
* 非事件處：(|\mu_t-\mu_{t-1}|\le\gamma)，所以 (g_t \le \gamma + \text{估計誤差})。
* 事件邊界處：(|\mu_t-\mu_{t-1}|\ge \Delta-\gamma)（保守界），所以 (g_t \ge \Delta-\gamma - \text{估計誤差})。
* 選 (\tau=\Delta/2) 並要求估計誤差小於 ((\Delta-2\gamma)/2) 即可分離。

這是最穩、最不容易「證不動」的一段。

---

### Theorem 2（端到端打分誤差界：總 token 預算可控）

在 (|w|\le 1) 下，若對每個 (t\in\widehat{T}) 取
[
m ;\ge; C_2\frac{\sigma^2}{\varepsilon^2}\log\frac{k}{\eta},
]
並將非事件時間用每段背景代表（每段再用 (m_0=O(\sigma^2/\varepsilon^2\log(1/\eta))) tokens 估計一次背景均值），則以機率至少 (1-\eta)：
[
|\hat s - s^\star|
;\le;
O!\left(\varepsilon\right)
]
且總 token 使用量
[
B ;=; rT ;+; mk ;+; m_0(#\text{背景段數})
;=;
\tilde O!\left(
T\frac{\sigma^2}{(\Delta-2\gamma)^2}\log\frac{T}{\eta}
;+;
(s+\log T)\frac{\sigma^2}{\varepsilon^2}\log\frac{s}{\eta}
\right).
]

**證明骨架：**

* 用 (|\langle w, \tilde\mu_t-\mu_t\rangle|\le |\tilde\mu_t-\mu_t|)。
* 將誤差分解為事件部分 + 背景部分，分別用均值估計集中界控制。
* 用 union bound 控制所有被用到的均值估計。

這定理的好處是：每一步都是標準集中工具，極少出現「假設牽強」。

---

### Theorem 3（下界：沒有足夠 token，必然會失敗）

在一個簡化但標準的子模型中（背景常數 (m_t\equiv 0)，事件為 (s) 個單點或短區間，事件時均值偏移為固定向量 (\delta)，(|\delta|=\Delta)，噪聲 (\sigma^2)-subGaussian），任何演算法若總讀取 tokens
[
B ;<; c\cdot \frac{\sigma^2}{\Delta^2}; s\log\frac{T}{s},
]
則存在一組分佈使其無法以機率超過 (2/3) 正確定位事件集合（或無法使 (|\hat s-s^\star|\le \varepsilon)）。

**證明骨架（Fano/多重假設檢定）：**

* 把每個可能事件集合 (E)（大小 (s)）視為一個假設，假設數量 (\binom{T}{s})。
* 每讀取一個 token 的 KL/互資訊增量上界為 (O(\Delta^2/\sigma^2))。
* 套 Fano：要把錯誤率壓到常數以下，需要總互資訊 (\Omega(\log \binom{T}{s})=\Omega(s\log(T/s)))，因此得出 (B) 下界。

這能防止你的工作被質疑「只是工程直覺」，因為你有 matching 的資訊論必要性。

---

## 6. 你寫論文時最不容易出錯的技術選擇

1. **先只做線性 head / Lipschitz loss 的理論**
   你把核心貢獻放在「稀疏事件結構 ⇒ 時空解耦壓縮近似最優」，不要一開始就把 Transformer attention 的誤差傳播扛進來。

2. **噪聲模型用 subGaussian（或 bounded）**
   集中不等式最成熟，審稿也最買單。

3. **事件條件用 (\Delta>2\gamma) 這種清楚可解釋的分離條件**
   它直觀對應「事件變化幅度必須大於背景漂移」。

4. **下界用 Fano（不要用太花的 lower bound 技巧）**
   Fano + KL bound 是最穩、最不容易被挑剔的路線。

---

## 7. 與 Ego4D / Ego-Exo4D 的對應（寫 Related/Discussion 時可直接用）

* Ego4D 長影片常見：大段背景視角維持（(\gamma) 小）、短時間 HOI 操作（(\Delta) 大、事件稀疏 (s) 小）。
* 你的理論直接說明：**為何 time-first（先找事件段）再在事件段做空間壓縮，在 token budget 下是（近）最優策略**。
* 對 NLQ/VQA，可把 (w) 或打分函數視為「query-conditioned 的線性探測器」作為第一步延伸，但不必在主定理硬塞進去（避免難度陡增）。