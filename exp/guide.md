下面是一個**最小（但能清楚支撐主張）**的 synthetic experiment 設計，直接對應你文中的模型、兩階段方法與兩個關鍵現象：**(i) margin (\kappa=\Delta-2\gamma) 的相變**、**(ii) drift–anchoring 的 (O(\gamma L)) bias/overhead trade-off**。 

---

## Synthetic data（只做 score-direction，一維就夠）

為了最小化，直接在投影後的標量 (y_{t,i}=\langle w,x_{t,i}\rangle) 上生成資料（等價於令 (d=1,w=1)），完全符合你文中的「只在 task-relevant direction 假設」設定。

**參數（建議固定一組即可）：**

* (T=2000)（時間長度）
* (N=256)（每個時間的空間 tokens 數）
* 事件段數 (s=5)，每段長度 (\ell=20)，事件段彼此間隔至少 100（避免黏在一起）
* 噪聲：(\varepsilon_{t,i}\sim\mathcal N(0,\sigma^2))，(\sigma=1)
* 背景漂移：(b_1=0)，(b_t=b_{t-1}+u_t)，其中 (u_t\in{+\gamma,-\gamma}) 隨機取（確保 (|b_t-b_{t-1}|\le \gamma)）
* 事件強度：事件段 (j) 取常數 shift (\beta_j\in{+\Delta,-\Delta})（隨機符號），且 (|\beta_j|\ge \Delta)

**生成：**
[
a_t = b_t + \mathbf 1{t\in E}\beta_t,\qquad
y_{t,i}=a_t+\varepsilon_{t,i}.
]
目標（ground truth）：
[
s^\star=\frac1T\sum_{t=1}^T a_t.
]
這正是你文中的稀疏事件 + 緩慢漂移背景、以及要保留全域平均 linear score 的最簡化版本。

---

## Methods（最小：1 個方法 + 2 個 baseline 就夠）

### Proposed：Time-screen → Space-refine + Background anchoring

照你文中兩階段：Stage I 每個時間抽 (r) 個 token 估 (\hat a_t)，用
[
g_t=|\hat a_t-\hat a_{t-1}|
]
做 boundary screening；Stage II 只在少量 anchor 上加抽樣，並每 (L) 步放一個 background anchor（估計漂移段）。

* threshold：synthetic 建議先用「oracle-friendly」(\tau=\Delta/2)（對應定理敘述，畫出最乾淨的相變），再加一個 ablation 用 MAD rule（可選）。
* refinement：event anchors 用 (m)，background anchors 用 (m_0)
* 估計量 (\hat s) 直接用你文中的 anchor averaging 版本（事件 anchor + 背景 anchor 填補）。

### Baseline 1：Uniform sampling（同 budget 平均灑在所有時間）

把總 budget (B) 平均分到每個時間 (q=\lfloor B/T\rfloor) 個 token，估 (\bar a_t) 後直接平均 (\hat s_{\text{uni}}=\frac1T\sum_t \bar a_t)。

### Baseline 2：Oracle segmentation（上界參考）

假設已知真實事件/背景分段（或真實 anchors），只做 Stage II 的 sampling/anchoring；用來顯示主要瓶頸是否真在 Stage I 的 screening。

---

## Metrics（兩個就夠）

1. **Score error**：(|\hat s - s^\star|)（主指標，直接對應你的目標）
2. **Boundary F1 / Recall（±1 容忍）**：把偵測到的邊界與真實 entry/exit 比對，容忍 (\pm 1) time-step（對應你文中的 pointwise (g_t) 討論）

每個設定重複 200 seeds，回報 mean±std。

---

## 兩張圖就能構成完整 “Experiment (Synthetic)” 小節

### Figure 1：驗證 (\kappa=\Delta-2\gamma) 的 screening 相變（對應 Thm.1）

固定 (\sigma=1,\gamma=0.05)，掃 (\Delta\in{0.12,0.16,0.20,0.28})，因此
[
\kappa=\Delta-2\gamma\in{0.02,0.06,0.10,0.18}.
]
對每個 (\kappa)，掃 (r\in{1,2,4,8,16,32,64})，畫：

* y 軸：boundary **Recall**（或 F1）
* x 軸：(r)
* 多條曲線：不同 (\kappa)

**你要在圖說點出的一句話：**
當 (\kappa) 變小，達到固定 recall（例如 0.95）所需的 (r) 會快速上升，符合 (r=\tilde\Theta(\sigma^2\kappa^{-2}\log T)) 的趨勢；(\kappa\le 0) 時會出現系統性失敗（miss 或 爆 false positives）。

---

### Figure 2：驗證 anchoring 的 (O(\gamma L)) drift bias 與 overhead trade-off（對應 Thm.2）

這張圖只要把統計誤差壓小，就能把 drift bias 單獨顯示出來：

* 固定一個「容易成功 screening」的設定：例如 (\Delta=0.25,\gamma=0.05\Rightarrow \kappa=0.15)，取足夠大 (r)（如 (r=32)）
* 把 (m,m_0) 設很大（如 512），讓 sampling variance 幾乎可忽略
* 掃 (L\in{5,10,20,40,80,160})

畫：

* y 軸：(|\hat s-s^\star|)
* x 軸：(L)

**你要在圖說點出的一句話：**
(|\hat s-s^\star|) 隨 (L) 近似線性增長（斜率 (\approx \gamma) 的量級），對應「用單一 anchor 代表長度 (L) 漂移段」的 deterministic bias (O(\gamma L))；同時在文內用一句話交代 anchor 數量 (|B_b|=\Theta(T/L))（overhead）隨 (L) 變大而下降。

> 若你想更 “最小”，Figure 2 甚至可以只畫 (|\hat s-s^\star|) vs (L)，不另外畫 overhead；overhead 直接寫在 caption 裡即可。

---

## 一段可直接貼進 paper 的 “Synthetic experiment” 文字骨架

* **Setup**：描述上述 (T,N,s,\ell,\sigma,\gamma,\Delta) 與生成式 (a_t=b_t+\mathbf1{t\in E}\beta_t)，目標 (s^\star=\frac1T\sum_t a_t)。
* **Compared methods**：Proposed（screen+refine+anchoring）、Uniform、Oracle segmentation。
* **Metrics**：score error、boundary recall/F1（±1）。
* **Findings**（兩句話收斂）：

  1. screening 需要的 (r) 對 (\kappa) 非常敏感，(\kappa\downarrow 0) 出現明顯 phase transition；
  2. anchoring 的誤差呈現 (O(\gamma L)) 的 drift bias，與 (\Theta(T/L)) 的 anchor overhead 形成經典 trade-off。

---

如果你願意再多加一個最小 ablation（不一定要），我建議只加這個：**把 threshold 從 oracle (\tau=\Delta/2) 換成 MAD rule**（你文中 Appendix 的想法），顯示結果趨勢一致但常數略差即可。
