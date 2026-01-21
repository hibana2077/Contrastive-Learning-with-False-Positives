# Contrastive Learning with False Positives

---

## One-liner（摘要第一句那種）

**我們在「語意翻轉增強」下，為線性 InfoNCE 表示學習給出一個由 BBP/spiked random matrix 決定的**兩側相變界**：上界保證任意近似極小值都會對齊語意子空間並支持小樣本線性 probe，下界則證明在另一側所有最優表示幾乎不含 label 資訊；在大負樣本或特定縮放下兩側界收斂，呈現近似 sharp 的臨界曲面。**

---

# Research Outline（可直接對應 paper 結構）

## 0. 目標與貢獻（你要賣的 3 個 punchlines）

### 0.1 問題核心

在 contrastive learning 中，augmentation 可能引入 **false positives**（正對其實不同語意）。你關心的是：**語意翻轉率 p 與 negatives 數 N、溫度 τ 如何共同決定「學到語意/學不到語意」的臨界現象**，而且要能做成 **兩側界**並在某些 regime 變得近似 sharp。

### 0.2 主要貢獻（建議寫成 Theorem 導向的 3 條）

1. **（可泛化的成功定理）**：在有效訊號強度 (\alpha(p)=(1-2p)^2\rho/\sigma^2) 超過 BBP 門檻時，**任何近似 InfoNCE 極小值**都會學到 (\mathrm{span}(\mu_{1:K}))，並給出 downstream 線性 probe 的小樣本複雜度。
2. **（可證明的失敗/不可辨識定理）**：在 (\alpha(p)) 低於另一門檻時，**所有 population 最優解皆「無語意」**（或更保守：所有只看二階統計的表示法都不可辨識），因此線性 probe 即使標記很多也受表示瓶頸限制。
3. **（兩側界 + 漸近 sharp）**：導出 (p_-(\cdot)\le p_+(\cdot)) 的兩側界；在 (N\to\infty) 且 (\tau) 合理縮放等 regime 下，兩側界收斂，得到 **asymptotically sharp** 的相變曲面。

---

## 1. Setting 與符號（讓證明能閉式化的最小假設）

### 1.1 資料：對稱正交 GMM

* (y\sim\mathrm{Unif}([K]))
* (x=\mu_y+\sigma z,; z\sim\mathcal N(0,I_d))
* (\mu_k\perp \mu_\ell) 且 (|\mu_k|^2=\rho)

### 1.2 Two-view augmentation：語意翻轉（false positives）

[
x^{(1)}=\mu_y+\sigma z_1,\quad
x^{(2)}=\mu_{\tilde y}+\sigma z_2
]
[
\Pr(\tilde y=y)=1-p,\quad \Pr(\tilde y\neq y)=p\ \text{且在 }[K]\setminus{y}\text{均勻}
]

### 1.3 表示：線性 + normalize

[
f_W(x)=\frac{Wx}{|Wx|}\in \mathbb S^{m-1},\quad W\in\mathbb R^{m\times d},\ m\ge K
]

### 1.4 目標：InfoNCE（N negatives、溫度 τ）

* cosine similarity（normalize dot product）
* negatives i.i.d. from marginal（或你也可以做 “in-batch negatives” 版本的 extension）

### 1.5 高維極限（可選但建議）

[
d,m,M\to\infty,\quad \gamma=d/m\ \text{固定},\quad M \text{為 unlabeled pairs 數}
]
這是把 BBP 做漂亮的關鍵。

---

## 2. 主量：有效訊號強度與「兩側界」的參數化

### 2.1 有效訊號強度（你已定義得很好）

[
\alpha(p):=(1-2p)^2\cdot \frac{\rho}{\sigma^2}
]
直覺：語意翻轉把跨 view 語意相關縮到 ((1-2p))，二階訊號強度縮到平方。

### 2.2 想得到的兩個 critical values（以 (\alpha) 表示）

* (\alpha_c^{\mathrm{BBP}}(\gamma,N,\tau,K))：成功側（足夠條件）
* (\alpha_c^{\mathrm{fail}}(\gamma,N,\tau,K))：失敗側（必要條件）

然後把它們轉回 (p_-)、(p_+)：
[
p_-=\tfrac12\Big(1-\sqrt{\alpha_c^{\mathrm{BBP}}\cdot \sigma^2/\rho}\Big),\quad
p_+=\tfrac12\Big(1-\sqrt{\alpha_c^{\mathrm{fail}}\cdot \sigma^2/\rho}\Big)
]

---

## 3. Population reduction：把 InfoNCE 化約到二階結構（整篇最關鍵的橋）

這一節要做到：**在你的對稱 GMM + 線性 + normalize 下，population InfoNCE 最優方向等價於某個加權 cross-covariance / CCA-like 的主特徵向量問題**（或至少給出 tight 上下界，足以導出門檻）。

### 3.1 你可以主張的形式（模板）

* 定義某個「有效 cross-covariance」矩陣（示意）
  [
  C := \mathbb E\big[\phi_{\tau,N}(x^{(1)},x^{(2)},{x^-*j})\cdot x^{(1)}(x^{(2)})^\top\big]
  ]
  其中 (\phi*{\tau,N}) 是由 softmax 權重誘導的標量權重（或向量權重），在某些 regime 可近似常數或只依賴內積。

* 然後證明：對線性 encoder（含 normalize），**最優的 row-space 由 (C) 的 top-(K) 子空間決定**（或至少：若 (C) 有 spike，則最優解會拾取 spike）。

### 3.2 兩個常見可走的「簡化極限」（用來把 (\phi) 變得可控）

1. **大 N + 合理 τ 縮放**：InfoNCE 接近某種 log-density ratio / softmax averaging，使權重集中，得到較乾淨的二階形式。
2. **(\tau\to 0)（max-approx）或 (\tau) 固定但用 Lipschitz 上下界**：把 log-sum-exp 夾在 max 和平均之間，導出可計算的 sandwich bounds。

> 你不一定要完全等價；COLT 風格可以接受「上下界足以推出相同 BBP 門檻」。

---

## 4. Spiked random matrix / BBP：導出成功側門檻 (\alpha_c^{\mathrm{BBP}})

### 4.1 你要分析的物件（經由上節化約）

通常會落到：某個 (m\times m) 或 (d\times d) 的樣本矩陣
[
\widehat C=\frac1M\sum_{i=1}^M u_i v_i^\top
]
其中 ((u_i,v_i)) 是由兩 view 的線性投影（或 whiten 後）得到；其期望包含 rank-(K) spike（對應 (\mu_{1:K})），其餘為 isotropic noise。

### 4.2 BBP 結論你要用的輸出（paper 需要的不是細節，而是「對齊量」）

* 存在臨界 (\alpha_c^{\mathrm{BBP}}(\gamma,\cdot))，當 spike 強度超過它：

  * top eigenvalue 脫離 bulk
  * top eigenvector 與真子空間的對齊（overlap）為正，且可給 closed-form（或下界）

你在 theorem A 裡要用的就是：
[
\sin^2\angle(\mathrm{row}(\hat W),\mathrm{span}(\mu_{1:K}))\le \text{(optimization + generalization errors)}
]

---

## 5. Empirical-to-population：近最小化器的一致性（COLT 味的核心）

你想要的是「不是只說存在一個好解」，而是：

> **任何** empirical InfoNCE 的 (\varepsilon_{\text{opt}})-近最小值，都必然靠近 population 最小值集合，進而對齊 spike 子空間。

### 5.1 可操作的做法（模板）

1. 先證明 InfoNCE loss（對線性+normalize）在參數集合上是 **（局部）Lipschitz** 或滿足某種截斷後的 Lipschitz。
2. 對假設類（例如 (|W|\le B)，或 row-orthonormal）做 uniform convergence：
   [
   \sup_W |\widehat{\mathcal L}(W)-\mathcal L(W)|\le \varepsilon_{\text{gen}}=\tilde O(\sqrt{d/M})
   ]
3. 得到「近最小化器 → 近 population 最小化器集合」：
   [
   \widehat{\mathcal L}(\hat W)\le \inf_W \widehat{\mathcal L}(W)+\varepsilon_{\text{opt}}
   \Rightarrow
   \mathcal L(\hat W)\le \inf_W \mathcal L(W)+(\varepsilon_{\text{opt}}+2\varepsilon_{\text{gen}})
   ]
4. 再把「population near-optimal → 子空間對齊」接上第 4 節的 BBP 幾何結論。

---

## 6. Downstream：線性 probe 小樣本複雜度（把 representation 轉成可量化任務）

這節的任務：把「對齊 (\mu) 子空間」變成「可分類」與 sample complexity。

### 6.1 你可以主張的典型 statement

* 若表徵空間中每一類的 mean 在某個 margin 下線性可分，則多類線性分類的泛化需要
  [
  n_{\text{lp}}=\tilde O\left(\frac{K}{\epsilon}\right)\ \text{或}\ \tilde O\left(\frac{K}{\epsilon^2}\right)
  ]
  取決於你用 0–1、margin bound、或 surrogate loss。

### 6.2 你需要的橋接 lemma（常見且乾淨）

* 表示對齊 ⇒ 類別在表示空間的中心距離 (\gtrsim) 某函數 (g(\alpha))
* 噪聲在表示空間的擾動 (\lesssim) 某函數 (h(\sigma,\alpha,m))
* 因此得到 margin，下游 sample complexity 就跟 margin 成反比。

---

## 7. 失敗側：不可辨識 / 無語意最優解（Theorem B）

你已經選了「夠強但更好證」的版本：**針對線性 InfoNCE（或更弱：所有二階統計表示）證明失敗**。

### 7.1 兩個可選強度（你可依證明難度挑）

**版本 B-strong（更像你原稿）**
若 (\alpha(p)\le(1-\delta)\alpha_c^{\mathrm{fail}})，則任何 population minimizer (W^\star) 都滿足
[
I(y;f_{W^\star}(x))=o(1)
\Rightarrow
\text{best linear probe acc}\le \tfrac1K+o(1)
]

**版本 B-safe（更穩，通常更快證）**
限制到「只依賴二階統計」的表示學習法：若 cross-covariance 的 spike 沒有脫離 bulk，則任何這類方法都無法 recover (\mathrm{span}(\mu_{1:K}))，因此表示不含可用 label 訊息。

> 實務上 B-safe 很 COLT：清楚、乾淨、審稿也比較容易買單。

### 7.2 你要證的核心事實（直覺版）

在低 (\alpha) 時，兩 view 的語意相關被 false positives 抹平，導致你化約出的二階物件（cross-covariance/CCA）只剩各向同性噪聲，top eigenspace 與真子空間不相關 ⇒ 表示對 label 幾乎獨立。

---

## 8. 兩側界收斂：asymptotically sharp 的 regime（你想保留的「像主定理」味道）

你要一個 section 專門說：

* 對一般有限 (N,\tau)：我們得到帶狀區 (p_-\le p\le p_+)
* 在某些 scaling：(\alpha_c^{\mathrm{BBP}}-\alpha_c^{\mathrm{fail}}\to 0)

### 8.1 典型可賣的 regime（你可挑一個做到漂亮）

1. **(N\to\infty) 且 (\tau) 隨 (\log N) 或緩慢縮放**：softmax 權重集中，使得 population reduction 的上下界貼近。
2. **(\tau\to 0) 的 max-limit**：InfoNCE 變成近似 hard negative mining，解析式可能更乾淨（但要小心優化地形變得尖）。
3. **固定 (K)，(\gamma) 固定，(M\to\infty)**：把 generalization 誤差消掉，剩下純粹 RMT 門檻。

---

# Experiments Outline（一定要做，不然理論像「盆栽」）

即使是理論 paper，也建議做最少但漂亮的合成實驗，專門驗證「兩側界 + 近似 sharp」。

## E1. 合成 GMM：相圖（主圖）

* 掃 (p\in[0,1/2])、掃 (N)、掃 (\tau)
* 指標：

  1. 表示與真子空間的 overlap（principal angles / subspace distance）
  2. 線性 probe accuracy vs (n_{\text{lp}})
  3. InfoNCE train loss（看是否有對應的曲率變化）
* 圖：

  * **Phase diagram heatmap**：x 軸 p、y 軸 N（或 (\tau)），顏色是 overlap/acc
  * 疊上理論 (p_-)、(p_+) 曲線（兩條線夾出帶狀區）

## E2. 驗證「兩側界收斂」的 regime

* 固定 (\gamma,K,\rho/\sigma^2)
* 讓 (N) 增大（同時用你理論建議的 (\tau(N))）
* 看 empirically：帶狀區寬度 (p_+-p_-) 是否縮小

## E3. 「只看二階統計」的 baseline（支撐 B-safe）

* 用 cross-covariance/CCA 的 top eigenspace 直接做表示（不用 InfoNCE）
* 對比：InfoNCE、CCA、random features
* 在低 (\alpha) 時應該全部接近 random；高 (\alpha) 時 InfoNCE/CCA 都起來（或 InfoNCE 因 ((N,\tau)) 更敏感）

## E4. 非單調性展示（你敘事裡的雙面刃）

* 固定 (p) 接近臨界，掃 (N)
* 觀察：太小 N 訊號不夠放大；太大 N 因 false negatives/同語意互斥干擾（視你的設定是否包含這機制）可能出現非單調
* 如果你目前模型 negatives 完全 i.i.d. from marginal，false negatives 的效應可能弱；你可以加一個更貼近 in-batch 的版本作 extension（但主定理先不背這個包袱）

---

# 圖表與消融清單（審稿人最愛）

1. **Fig.1**：模型示意圖（語意翻轉造成 false positives）
2. **Fig.2**：理論相圖 + 實驗相圖（疊 (p_-,p_+)）
3. **Fig.3**：對齊量（overlap）vs (\alpha(p))（顯示 BBP 形狀）
4. **Fig.4**：probe sample complexity 曲線（acc vs (n_{\text{lp}})）在成功/失敗兩側
5. **Ablation**：

   * 固定 (N) 改 (\tau)
   * 固定 (\tau) 改 (N)
   * 改 (\gamma=d/m)
   * 改 (K)（先固定 K 最乾淨，再補一張 K 變動圖）

---

# 可能的 Lemma / Theorem 清單（寫作時的骨架）

你可以把整篇拆成 6 個可交付的 lemma 模塊：

1. **Lemma (Population reduction)**：(\mathcal L_{\text{InfoNCE}}(W)) 與某個二階目標（CCA-like）等價或可 sandwich。
2. **Lemma (Spike strength)**：在語意翻轉下，該二階目標的 spike 強度正比於 ((1-2p)^2\rho/\sigma^2)（或同量級）。
3. **Theorem (BBP success)**：若 (\alpha>\alpha_c^{\mathrm{BBP}})，則估計的 top 子空間與 (\mathrm{span}(\mu_{1:K})) 有非零對齊且可定量。
4. **Theorem (Uniform convergence + near-minimizers)**：empirical (\varepsilon_{\text{opt}})-近最小值 ⇒ population near-optimal。
5. **Theorem A (End-to-end success)**：把 3+4 組起來，得到「任意近最小化器」對齊 + 線性 probe 小樣本。
6. **Theorem B (Failure / identifiability)**：若 (\alpha<\alpha_c^{\mathrm{fail}})，則 population 最優解無語意 / 二階法不可辨識。

最後一節再放：

7. **Corollary (Two-sided bounds & asymptotic sharpness)**：定義 (p_-,p_+)，並給出收斂條件。

---

# Extensions（放到 discussion / appendix，讓故事更大但不拖主證明）

1. **非正交 (\mu_k)**：允許類別中心有 coherence，門檻改成依賴 Gram matrix 的譜。
2. **非均勻 flip（class-dependent p）**：得到不對稱相圖，可能更貼近 real augmentation。
3. **in-batch negatives / false negatives 明確化**：把 (N) 的雙面刃做得更真，可能出現更強的非單調。
4. **非線性但「只看二階」的表示族**：把失敗側推廣到更大類（kernel CCA / random features）。
5. **optimization dynamics**：在成功區證 GD 會進入對齊 basin（可選，有力但工作量大）。

---

# 你可以怎麼在摘要/引言賣（超精準 COLT 口吻）

* **現象**：false positives 會抹除跨 view 語意相關，導致對比學習在某個污染率後學不到語意。
* **理論**：在線性 normalized encoder + 對稱 GMM 下，InfoNCE 的學習行為由一個 spiked random matrix 的 BBP 相變控制。
* **結果**：我們給出兩側界：一側保證任意近最小化器對齊語意子空間並支援小樣本 probe；另一側則不可辨識，表示幾乎不含 label 訊息；在大負樣本/特定縮放下兩界收斂，呈現近似 sharp 臨界曲面。