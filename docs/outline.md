# Contrastive Learning with False Positives

## Main Results

### Setting（簡述，讓定理自洽）

令語意子空間為
$$
\mathcal S:=\mathrm{span}(\mu_1,\dots,\mu_K)\subset\mathbb R^d,\qquad m\ge K.
$$
資料生成與兩視角增強如你 outline：兩 view 的語意一致性會以「語意翻轉率」$p\in[0,\tfrac12]$ 被污染；表示學習用 **線性 + normalize** 編碼器
$$
f_W(x)=\frac{Wx}{|Wx|}\in\mathbb S^{m-1},\qquad W\in\mathbb R^{m\times d},
$$
並最小化（population / empirical）InfoNCE loss $\mathcal L(W)$ / $\widehat{\mathcal L}(W)$，其中 negatives 數為 $N$、溫度為 $\tau$。

---

### 1) 有效訊號強度與兩側界參數化

**Definition 1 (Effective signal strength).** 定義有效訊號強度
$$
\alpha(p):=(1-2p)^2\cdot \frac{\rho}{\sigma^2}.
$$
直覺是：語意翻轉把跨 view 的一階相關縮到 $(1-2p)$，而你最後用到的 spike 強度是二階量，所以進入 $(1-2p)^2$（你 outline 也已固定這個定義）。 

接著定義「成功側門檻」與「失敗側門檻」。因為你後面要走 *InfoNCE $\leftrightarrow$ CCA-like sandwich*，所以最乾淨的寫法是把門檻定義在那個 **reduced 二階 spiked random matrix** 上（細節留到 Lemma/Section 3–4）。

**Definition 2 (BBP success threshold).** 令 $\widehat{\mathbf C}$ 為你在 Population reduction 後得到的 $m\times m$（或 $d\times d$）二階樣本矩陣，其 **null model**（$\alpha=0$）譜在高維極限 $(d,m,M\to\infty,\ \gamma=d/m)$ 收斂到某個 bulk，且上緣為 $\lambda_+(\gamma,N,\tau)$。定義
$$
\alpha_c^{\mathrm{BBP}}(\gamma,N,\tau,K)
:=\inf\Big\{\alpha>0:\ \text{top-}K\ \text{eigenvalues of }\widehat{\mathbf C}\ \text{脫離 bulk，且估計子空間對 }\mathcal S\text{有非零 overlap}\Big\}.
$$
（等價地，你也可用「第一個 outlier 出現」或「top eigenspace 與 $\mathcal S$ 的主角度收斂到 $<\pi/2$」來定義。）

**Definition 3 (Failure / non-identifiability threshold).** 同樣在相同極限下，定義
$$
\alpha_c^{\mathrm{fail}}(\gamma,N,\tau,K)
:=\sup\Big\{\alpha\ge 0:\ \text{top-}K\ \text{eigenspace of }\widehat{\mathbf C}\ \text{與 }\mathcal S\ \text{漸近正交（overlap}\to 0)\Big\}.
$$
由 sandwich 結構，通常 $\alpha_c^{\mathrm{fail}}\le \alpha_c^{\mathrm{BBP}}$（因此形成帶狀區域）。 

最後把 $\alpha$-門檻轉回 $p$-門檻：

**Definition 4 (Two-sided phase boundaries).** 定義
$$
p_-:=\frac12\Big(1-\sqrt{\alpha_c^{\mathrm{BBP}}\cdot \sigma^2/\rho}\Big),\qquad
p_+:=\frac12\Big(1-\sqrt{\alpha_c^{\mathrm{fail}}\cdot \sigma^2/\rho}\Big),
$$
並理解為截到 $[0,\tfrac12]$（若根號外超出範圍就投影回去）。 

---

### 2) Theorem A：任意近最小化器 $\Rightarrow$ 對齊語意子空間 $\Rightarrow$ 小樣本 probe

先定義「近最小化器」：對給定 $\varepsilon_{\mathrm{opt}}\ge 0$，稱 $\widehat W$ 是 empirical $\varepsilon_{\mathrm{opt}}$-near-minimizer 若
$$
\widehat{\mathcal L}(\widehat W)\le \inf_W \widehat{\mathcal L}(W)+\varepsilon_{\mathrm{opt}}.
$$

再定義一個可量化的對齊指標（你之後可以換成 principal angles / projection distance 都行）：
$$
\mathrm{Align}(\widehat W)
:=\frac1K\big| \mathbf P_{\mathrm{row}(\widehat W)},\mathbf P_{\mathcal S}\big|_F^2\in[0,1],
$$
其中 $\mathbf P_{\mathcal U}$ 表示子空間 $\mathcal U$ 的正交投影矩陣。

**Theorem A (End-to-end success for near-minimizers).**
固定 $K$ 並令 $(d,m,M)\to\infty$ 且 $\gamma=d/m\to\gamma_0\in(0,\infty)$。存在常數 $c,C>0$ 使得：若污染率 $p$ 滿足
$$
\alpha(p)\ \ge\ (1+\delta)\alpha_c^{\mathrm{BBP}}(\gamma_0,N,\tau,K)
$$
對某個 $\delta\in(0,1)$，且 unlabeled pair 數 $M$ 足夠大使 uniform convergence 誤差 $\varepsilon_{\mathrm{gen}}=\sup_W|\widehat{\mathcal L}(W)-\mathcal L(W)|$ 滿足 $\varepsilon_{\mathrm{gen}}\le c\delta$，那麼以高機率（隨 $M$ 增大趨近 1）對所有 empirical $\varepsilon_{\mathrm{opt}}$-near-minimizers $\widehat W$ 都有
$$
\mathrm{Align}(\widehat W)\ \ge\ 1- C\cdot \frac{\varepsilon_{\mathrm{opt}}+\varepsilon_{\mathrm{gen}}}{\delta},
$$
亦即 **任意近最小化器都必然對齊語意子空間 $\mathcal S$**。

進一步，令表示為 $r=f_{\widehat W}(x)$。若 $\mathrm{Align}(\widehat W)\ge a$（例如上式給出 $a=1-o(1)$），則存在一個線性 probe（多類線性分類器）使得：用 $n$ 個標記樣本（可取為每類 $n/K$ 或總數 $n$，視你後面採用的泛化界）即可達到錯誤率 $\le \eta$，其中樣本複雜度可寫成
$$
n\ \gtrsim\ \widetilde{\mathcal O}\left(\frac{K+\log(1/\eta)}{\mathrm{Margin}(a,\rho/\sigma^2)}\right),
$$
且 margin 是一個隨 $\sqrt{a}$ 與 $\sqrt{\rho/\sigma^2}$ 單調增加的函數（你在 Section 6 用「對齊 $\Rightarrow$ 類中心分離」把它具體化）。

> 這個定理正是你說的：「任意近最小化器 → 對齊語意子空間 → 小樣本 probe」。 

---

### 3) Theorem B：低於門檻 $\Rightarrow$ minimizer 無語意 / 二階法不可辨識

你 outline 裡說要用 safe 版（「二階統計不可辨識」）會更穩，我這裡就把 theorem 寫成那個版本；你之後若想強化成 mutual information / Bayes acc 也可以在 appendix 加強。

**Theorem B (Failure / non-identifiability below threshold).**
在相同高維極限下，存在常數 $c>0$ 使得：若
$$
\alpha(p)\ \le\ (1-\delta)\alpha_c^{\mathrm{fail}}(\gamma_0,N,\tau,K)
$$
則下列敘述成立：

1.（二階不可辨識）對任何僅依賴你 reduction 後二階矩陣 $\widehat{\mathbf C}$（或其等價二階統計，如 CCA/cross-covariance 類）所構造的語意子空間估計 $\widehat{\mathcal S}$，其與真實 $\mathcal S$ 的 overlap 皆滿足
$$
\big|\mathbf P_{\widehat{\mathcal S}},\mathbf P_{\mathcal S}\big|_F^2 = o(1).
$$
2.（表示無語意：對線性 probe 而言）因此對任何以此類二階法得到的表示 $r$，最佳線性 probe 的分類表現漸近上與 random guess 無異（例如 top-1 accuracy $\le 1/K+o(1)$）。

> 這對應你要的：「低於門檻 → minimizer 無語意 / 二階法不可辨識」。 

---

### 4) Corollary：兩側界在特定 regime 下收斂（asymptotically sharp）

**Corollary (Two-sided bounds and asymptotic sharpness).**
令 $(p_-,p_+)$ 由 Definition 4 給出，則對所有 $(\gamma,N,\tau,K)$：
$$
p_-(\gamma,N,\tau,K)\ \le\ p_+(\gamma,N,\tau,K),
$$
且形成一個「帶狀區域」：

* 若 $p<p_-$，由 Theorem A 可保證學到語意（對齊 + 小樣本 probe）。
* 若 $p>p_+$，由 Theorem B 可保證不可辨識（表示無語意）。

此外，在你指定的 sharp regime（例如 $N\to\infty$ 且 $\tau=\tau_N$ 以使 population reduction 的上下夾逼常數收斂、sandwich gap $\to 0$），有
$$
\alpha_c^{\mathrm{BBP}}(\gamma,N,\tau_N,K)-\alpha_c^{\mathrm{fail}}(\gamma,N,\tau_N,K)\to 0,
$$
因此
$$
p_+ - p_- \to 0,
$$
兩側界收斂到同一個臨界污染率 $p_c$，得到 **asymptotically sharp 的相變曲面**。 

## Proof Overview

### Setting 與主參數（提醒讀者我們要證什麼）

語意子空間 $\mathcal S:=\mathrm{span}(\mu_1,\dots,\mu_K)\subset\mathbb R^d$，資料為對稱正交 GMM：
$$
y\sim \mathrm{Unif}([K]),\qquad x=\mu_y+\sigma z,\ z\sim\mathcal N(0,I_d),\qquad \mu_k\perp \mu_\ell,\ |\mu_k|^2=\rho.
$$
兩視角增強含 false positives（語意翻轉率 $p$）：
$$
x^{(1)}=\mu_y+\sigma z_1,\qquad x^{(2)}=\mu_{\tilde y}+\sigma z_2,
$$
其中 $\Pr(\tilde y=y)=1-p$，$\Pr(\tilde y\neq y)=p$（在 $[K]\setminus{y}$ 均勻）。

表示學習用線性 + normalize 編碼器
$$
f_W(x)=\frac{Wx}{|Wx|}\in \mathbb S^{m-1},\qquad W\in\mathbb R^{m\times d},\ m\ge K,
$$
並最小化 InfoNCE（$N$ 個 negatives、溫度 $\tau$）的 population / empirical loss：$\mathcal L(W)$ 與 $\widehat{\mathcal L}(W)$。

我們用你定義的有效訊號強度（主參數）
$$
\alpha(p):=(1-2p)^2\cdot \frac{\rho}{\sigma^2},
$$
並以此刻畫成功/失敗門檻與兩側相變界。

---

## (L1) Lemma：Population reduction（InfoNCE 被 CCA-like 目標夾住）

**目標：** 把「非線性的 InfoNCE」化約成「二階結構（cross-cov/CCA-like）」的最大化問題（或至少給 tight sandwich，上下界有同一個主特徵空間）。

### 形式（可放在 lemma 的 statement）

令 $r=f_W(x^{(1)})$、$r^+=f_W(x^{(2)})$、$r_j^-=f_W(x_j^-)$。InfoNCE 的單樣本 loss 可寫成
$$
\ell(W)= -\log \frac{\exp(\langle r,r^+\rangle/\tau)}
{\exp(\langle r,r^+\rangle/\tau)+\sum_{j=1}^N \exp(\langle r,r_j^-\rangle/\tau)}.
$$
定義一個由 softmax 權重誘導的「有效 cross-covariance」：
$$
\mathbf C(W):=\mathbb E\big[\phi_{\tau,N}(r,r^+,{r_j^-})\cdot \widetilde x^{(1)}(\widetilde x^{(2)})^\top\big],
$$
其中 $\widetilde x^{(v)}$ 表示（可選）經過 centering/whitening 的 view-$v$ 特徵（技術上用來把噪聲部分變成近似各向同性），$\phi_{\tau,N}$ 是一個非負且（在你指定的 regime）主要依賴內積的權重。

**Lemma（sandwich 版）**：存在只依賴 $(N,\tau)$ 的單調函數 $\Psi_-,\Psi_+$ 與常數項 $c_-,c_+$，使得對所有允許的 $W$，
$$
c_- - \Psi_-!\Big(|\mathbf C(W)|*{\mathrm{CCA}}\Big)\ \le\ \mathcal L(W)\ \le\ c*+ - \Psi_+!\Big(|\mathbf C(W)|*{\mathrm{CCA}}\Big),
$$
其中 $|\cdot|*{\mathrm{CCA}}$ 可取為（top-$K$）奇異值和、或 $|\cdot|_F^2$、或 canonical correlations 的總和（你選一個最終好推 RMT 的版本即可）。因此「最小化 InfoNCE」被化約成「最大化一個 CCA-like 二階量」。

### 證明想法（3 行直覺 + 2 個常用不等式）

1. 用 log-sum-exp 的上下界（或 Jensen/對偶）把 $\ell(W)$ 夾在「正樣本內積」與「負樣本 log-partition」之間；關鍵是 $\tau$ 與 $N$ 決定權重集中程度。
2. 在你的對稱 GMM + 各向同性噪聲下，negatives 的貢獻在 population 近似只依賴「內積分佈」的標量，主要把目標變成「提升 $\mathbb E[\langle r,r^+\rangle]$ 或其凸函數」。
3. 對線性+normalize，$\mathbb E[\langle r,r^+\rangle]$（或其可控 surrogate）再等價於 whiten 後的 cross-covariance 的（top-$K$）譜量，得到 CCA-like reduction。

---

## (L2) Lemma：Spike strength（rank-$K$ spike 強度 $\propto \alpha(p)$）

**目標：** 把 false positives 的污染率 $p$，轉成 reduced 二階矩陣裡「spike 強度」的解析縮放。

### 核心計算（語意相關被縮小）

在 semantic 子空間 $\mathcal S$ 上，兩 view 的語意相關（在適當 centering/whitening 後）會被一個係數 $\kappa(p)$ 縮小，從而二階 spike 強度縮到 $\kappa(p)^2$ 的量級。你在主敘事中取的有效訊號強度是
$$
\alpha(p)=(1-2p)^2\cdot \frac{\rho}{\sigma^2}.
$$
（備註：若要對一般 $K$ 寫得更精確，常見會得到 $\kappa(p)=(1-p)-\frac{p}{K-1}$，而在 $K=2$ 時即化為 $1-2p$；但在 overview 裡直接以你定義的 $\alpha(p)$ 作為 spike 參數即可。）

### Lemma（你之後接 BBP 用的輸出）

令 reduction 後的關鍵樣本矩陣（例如 cross-cov/whitened cross-cov）為
$$
\widehat{\mathbf C}=\frac1M\sum_{i=1}^M u_i v_i^\top,
$$
則其期望可分解為
$$
\mathbb E[\widehat{\mathbf C}]=\underbrace{\mathbf C_0}*{\text{isotropic / null part}}+\underbrace{\theta(p)\cdot \mathbf P*{\mathcal S}}_{\text{rank-}K\ \text{spike}},
$$
且 spike 係數滿足
$$
\theta(p)\asymp \alpha(p)\quad(\text{至多差一個只依賴 }K,N,\tau\text{ 的常數因子}).
$$
這一步把「augmentation 污染」變成「spiked random matrix 的 spike 強度」。

---

## (T1) Theorem：BBP success（特徵值脫離 bulk + overlap）

**目標：** 一旦 spike 強度超過臨界值，就能保證（i）top 特徵值/奇異值脫離 bulk，（ii）top 子空間與真實 $\mathcal S$ 有正的 overlap。

### 典型結論（你需要的「幾何輸出」）

在高維極限 $d,m,M\to\infty$，$\gamma=d/m\to\gamma_0$ 下，存在臨界
$$
\alpha_c^{\mathrm{BBP}}=\alpha_c^{\mathrm{BBP}}(\gamma_0,N,\tau,K),
$$
使得若 $\alpha(p)>(1+\delta)\alpha_c^{\mathrm{BBP}}$，則 $\widehat{\mathbf C}$ 的 top-$K$ 譜會產生 outliers，且估計的 top-$K$ 子空間 $\widehat{\mathcal S}$ 滿足
$$
\mathrm{Overlap}(\widehat{\mathcal S},\mathcal S)
:=\frac1K|\mathbf P_{\widehat{\mathcal S}}\mathbf P_{\mathcal S}|_F^2\ \ge\ c(\delta)>0.
$$
（很多 spiked model 甚至可給 closed-form overlap；在 overview 中你只需要「有 gap $\Rightarrow$ 有非零對齊」這個方向即可。）

---

## (T2) Theorem：Uniform convergence + near-minimizers（empirical 近最小 $\Rightarrow$ population 近最小）

**目標：** 不是只證「存在好解」，而是證「任何 empirical 的近最小化器都會落在 population 最小化器附近」，從而被迫對齊 spike 子空間。

### 兩步結構

1. **Uniform convergence**：在合適的假設類（例如 row-orthonormal 或 $|W|*{\mathrm{op}}\le B$）上，InfoNCE loss 對 $W$ 的波動可控（透過 Lipschitz + covering/Rademacher），得到
   $$
   \varepsilon*{\mathrm{gen}}:=\sup_W\big|\widehat{\mathcal L}(W)-\mathcal L(W)\big|
   \ \le\ \widetilde O\Big(\sqrt{\tfrac{d}{M}}\Big).
   $$
2. **Near-minimizer transfer**：若 $\widehat W$ 是 empirical $\varepsilon_{\mathrm{opt}}$-near-minimizer，
   $$
   \widehat{\mathcal L}(\widehat W)\le \inf_W \widehat{\mathcal L}(W)+\varepsilon_{\mathrm{opt}},
   $$
   則必有
   $$
   \mathcal L(\widehat W)\le \inf_W \mathcal L(W)+(\varepsilon_{\mathrm{opt}}+2\varepsilon_{\mathrm{gen}}).
   $$


### 為何這會導致「子空間對齊」？

當 BBP 條件成立時，population reduction 的 CCA-like 目標在 $\mathcal S$ 上有**嚴格譜間隙**：偏離 $\mathcal S$ 的子空間會付出至少 $\Omega(\delta)$ 的 population 代價。結合上式就得到「近最小 $\Rightarrow$ 不能偏離太多」，把 loss gap 轉成對齊量下界（下一段 Theorem A 就是這一步的整理）。

---

## (T3) Theorem A：End-to-end success（任意近最小化器都會學到語意）

把 (L1)+(L2)+(T1)+(T2) 串起來：

* (L1) 告訴你 InfoNCE 的最優方向由某個二階矩陣控制；
* (L2) 告訴你這個二階矩陣是 spiked，且 spike 強度 $\asymp \alpha(p)$；
* (T1) 告訴你 $\alpha(p)$ 超過 BBP 門檻會產生 outlier + overlap；
* (T2) 把 overlap 從 population 推到 empirical 近最小化器。

因此得到你在 `outline.md` 中寫的型態：若 $\alpha(p)\ge (1+\delta)\alpha_c^{\mathrm{BBP}}$ 且 $M$ 夠大，則對所有 empirical near-minimizers $\widehat W$，
$$
\mathrm{Align}(\widehat W)
:=\frac1K\big|\mathbf P_{\mathrm{row}(\widehat W)}\mathbf P_{\mathcal S}\big|*F^2
\ \ge\ 1- C\cdot \frac{\varepsilon*{\mathrm{opt}}+\varepsilon_{\mathrm{gen}}}{\delta}.
$$
接著「對齊 $\Rightarrow$ 類中心分離 $\Rightarrow$ 線性 probe 小樣本」屬於標準橋接（margin bound / 多類線性分類泛化），因此得到 downstream sample complexity。

---

## (T4) Theorem B：Failure（safe 版：二階統計不可辨識）

**目標：** 在另一側，給一個更穩、更好證的失敗定理：只要 spike 沒有脫離 bulk，任何「只依賴二階統計」的方法都無法 recover $\mathcal S$。

當
$$
\alpha(p)\le (1-\delta)\alpha_c^{\mathrm{fail}}(\gamma_0,N,\tau,K),
$$
則 reduction 後的 $\widehat{\mathbf C}$ 在 null/bulk 中沒有資訊性 outlier，top-$K$ eigenspace 與 $\mathcal S$ 漸近正交：
$$
|\mathbf P_{\widehat{\mathcal S}}\mathbf P_{\mathcal S}|_F^2=o(1).
$$
因此任何以這類二階物件（cross-cov/CCA/其等價統計）構造的表示，都不含可用的語意方向，最佳線性 probe 的表現漸近上不優於 random guess（例如 top-1 accuracy $\le 1/K+o(1)$）。

---

## Corollary：兩側界與（某些 regime 的）sharpness

定義兩側相變邊界（把 $\alpha$ 門檻換回 $p$）：
$$
p_-=\frac12\Big(1-\sqrt{\alpha_c^{\mathrm{BBP}}\cdot \sigma^2/\rho}\Big),\qquad
p_+=\frac12\Big(1-\sqrt{\alpha_c^{\mathrm{fail}}\cdot \sigma^2/\rho}\Big),
$$
則有帶狀區域 $p_-\le p\le p_+$：

* $p<p_-$ 時由 Theorem A 保證成功；
* $p>p_+$ 時由 Theorem B 保證失敗。

最後，在你強調的 sharp regime（例如 $N\to\infty$ 且 $\tau=\tau_N$ 使 population reduction 的 sandwich gap $\to 0$），兩個門檻收斂：
$$
\alpha_c^{\mathrm{BBP}}(\gamma,N,\tau_N,K)-\alpha_c^{\mathrm{fail}}(\gamma,N,\tau_N,K)\to 0
\quad\Rightarrow\quad
p_+-p_-\to 0,
$$
因此得到近似 sharp 的臨界曲面。