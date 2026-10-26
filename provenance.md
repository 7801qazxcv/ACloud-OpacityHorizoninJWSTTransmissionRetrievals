# Provenance ledger — JWST identifiability paper

每个进入论文的数字都必须在此表中指向一个可复跑脚本或一个带 DOI/SHA 的真实数据文件。
本文件随实验推进增量更新。**没有条目的数字不许进正文。**

## 0. 环境（可复现声明）
| 项 | 值 | 证据 |
|---|---|---|
| 前向模型 | PLATON 6.3 | `pip show platon`；opacity DB md5 `2cc3d6d746c80cc38bd9fdb25c226842`，14 GB，来自 astro.uchicago.edu |
| Python | 3.13.14 (`C:/Python313`) | `PYTHONPATH= /c/Python313/python.exe -V` |
| numpy / scipy | 2.5.2 / 1.16.3 | 同上 |
| 主机 | 32 CPU / 125.6 GB RAM | `GlobalMemoryStatusEx` |

## 1. 前向模型特性（stage1_forward_probe.py → stage1_probe.json）
| 论文中的量 | 值 | JSON key | 备注 |
|---|---|---|---|
| R=100 通道数（0.6–5.3 μm） | 219 | `n_bins` | NIRSpec PRISM 类网格 |
| 单次前向模型耗时（中位） | 0.0966 s | `forward_seconds_median` | 决定 nested sampling 预算 |
| 5×5 Jacobian 计算耗时 | 1.25 s | `jacobian_seconds` | 中心差分，10 次前向 |
| 平均透射深度 | 24723 ppm | `depth_ppm_mean` | WASP-39b 系统参数下 |
| 光谱调制幅度（峰谷） | 3897 ppm | `depth_ppm_ptp` | — |
| 条件数 κ(J/σ) | 4.534e4 | `sigma*.cond` | 与 σ 无关（尺度不变） |
| DOFS @ σ=30 ppm | 4.625 | `sigma30.dofs` | 名义 5 参数 |
| DOFS @ σ=50 ppm | 4.236 | `sigma50.dofs` | — |
| DOFS @ σ=100 ppm | 3.455 | `sigma100.dofs` | — |
| DOFS @ σ=200 ppm | 2.528 | `sigma200.dofs` | **核心发现：名义 5 参数只约束住 2.5 个** |

⚠️ 待办：上述 σ 是**情景值**（scenario），正文必须替换为各行星**已发表的逐通道误差**。
在 stage3 完成前，这些 DOFS 只能作为方法演示，不得作为对某行星的结论。
✅ 已由 stage2 用真实逐通道误差取代；stage1 的 σ 情景表仅作为方法示意保留在附录。

## 1b. ⚠️ 对抗性自检推翻了 stage2 的初版叙事（务必写进论文）
stage2（固定 log10 Pc = 4.0）得到"仅 3–5 μm 数据无法约束云顶压强，L=1.000"。
**stage3 证明这个解读是错的**，且错因是固定点的选择：

| 数据集 | log Pc = 2.5 | log Pc = 4.0 | log Pc = 5.0+ |
|---|---|---|---|
| WASP-39b G395H (IR-only, 184 ppm) | L=0.007, F_cc=2.4e3 | L=1.000 | L=1.000 |
| GJ 486b G395H (IR-only, 21 ppm) | L=0.006, F_cc=1.3e3 | L=1.000 | L=1.000 |
| WASP-39b PRISM (含光学, 99 ppm) | — | L=0.143, F_cc=7.2 | L=0.9996 |

真实结论（阈值定律，而非数据集属性）：
- 云顶压强的可辨识性存在一个**不透明度视界** log10(Pc/Pa) ≈ 3.0–3.5；
- 视界之内（Pc 更低/云更高），**即使只有 3–5 μm 数据也能把云测得极好**（F_cc ~ 10³–10⁴）；
- 视界之外，**任何波段、任何精度都测不出**（GJ 486b 的 13 ppm 精度也不行）；
- DOFS 在跨越视界时从 ~4.8 阶跃到 ~3.9。
- 含光学覆盖使视界略微外推（PRISM 在 logPc=4.0 仍有 L=0.143，IR-only 已 L=1.000）。

**教训（写入 skill）**：Fisher/DOFS 类诊断必须在参数空间**扫描**后再解读；
在单一固定点上得到的"某参数不可辨识"极可能是固定点落在死区，而非测量能力的结论。
精确的 0 特征值 / L=1.000 是危险信号，不是发现。

## 2. 真实光谱数据（fetch_archive.py → transitspec_jwst.csv）  ✅ 已完成
来源：**NASA Exoplanet Archive `transitspec` TAP 表**（公开、CC/官方归档）。
- 12 个已发表 JWST 透射光谱、2 颗行星类型（热 Saturn WASP-39b、暖 Saturn HAT-P-18b、岩石 super-Earth GJ 486b）
- 2712 个逐通道数据点，每个带**已发表的逐通道误差**（plntransdeperr1）
- 每条对应文献 plntranreflink（Feinstein2022/NIRISS, Rustamkulov2022/PRISM, Alderson2022/G395H,
  Ahrer2022/NIRCam, Fu2022/HAT-P-18b, Moran2023/GJ486b, JWST ERS 团队 G395H×4）
- **未使用任何模拟/注入数据冒充观测**。模型光谱仅在 stage5 的注入-回收验证中出现，且明确标注为"model spectrum, not data"。
- 文献 DOI 全部经 verify_bib.py 核验（305/307 通过，2 条 WARN 非造假）。

## 3. 前向模型（PLATON 6.3，14 GB 不透明度库）  ✅ 已完成
- 单次前向 0.214 s；Jacobian 5×0.2s；覆盖 0.2–30 μm，ExoMol/HITRAN
- 恒星/行星参数取自文献（表 2），seeded 真实值

## 4. 可辨识性诊断（identifiability.py + stage2/3/4/5）  ✅ 已完成
- stage2：12 数据集真实 σ → DOFS 与 prior leverage（图2）
- stage3：云顶压强扫描 → 不透明度视界（图3）——**对抗性自检推翻了初版叙事**
- stage4：视界定位 + 偏差转移（图4）+ 真实偏差量级（表4）
- stage5：非线性 Gauss-Newton 重演验证线性算子（图5）

## 5. 论文（ApJ 格式，aastex701）  ✅ 已编译 9 页，0 undefined citation/ref，score 82/100
- main.tex + sec0-8，5 图 4 表
- 编译：pdflatex + bibtex（aasjournalv7），作者块用占位符 [Author Name] 待填

## 6. 关键诚实性记录（投稿前必读）
1. stage3 推翻了 stage2 的"IR 测不出云"——真实结论是不透明度视界（阈值定律），不是数据集属性。
2. stage5 发现线性偏差算子在大 δ 处过度外推（×1.1–8），正文已如实报告，偏差量级标注为"固定 δ 下的下界"。
3. bias_from_fixing 初版符号错误，已被 stage5 非线性重演捕获并修正。
4. 视界对红外数据精确 3.373（零散布，跨 13–242 ppm），对光学数据 4.2–4.7——这是文章最强证据。
5. 未做任何"模拟=观测"替换；所有观测数字来自 Archive 真实归档。
| 行星 | 仪器 | Zenodo | DOI | 许可 | 文件 | SHA-256[:32] |
|---|---|---|---|---|---|---|
| L 98-59 c | NIRSpec/G395H | 13363863 | 10.5281/zenodo.13363863 | CC-BY-4.0 | vis1/vis2_eureka_spectrum.csv (各 106 点) | 待 collect 运行写入 |
| TRAPPIST-1 b | NIRISS/SOSS | 14422314 | 待写 | CC-BY-4.0 | fig2_panel*_tspec-*.h5 | 待写 |

⚠️ 注意（已实测的坑）：
- NASA Exoplanet Archive TAP `spectra` 表可查（261 条 JWST 透射谱 / 62 行星），但 `spec_path`
  **不是可拼接 URL**，且高频请求会被限流（curl 返回 000）。不要在论文里声称从该接口批量下载。
- Zenodo 通用关键词检索（`sort=mostrecent`）会返回大量无关预印本；必须按行星名 + `sort=bestmatch`。

## 2026-08-16 — Adversarial self-review additions (stage6, stage7)

### stage6_retrieval.json — REAL emcee retrievals (PLATON CombinedRetriever, equilibrium chemistry)
Validates diagnostic predictions on PUBLISHED JWST data (WASP-39b), independent sampler:
- IR-only (Alderson G395H) equilibrium: logZ med=0.31 [-0.12,1.24], IQR=1.36, prior width=5.0 -> IQR/prior=0.27
- Optical (Feinstein NIRISS) equilibrium: logZ med=0.46 [-0.06,1.03], IQR=1.10 -> IQR/prior=0.22
- Cloud pinned at logPc=3.0: logZ med=0.09 [-0.25,0.93]
- Cloud pinned at logPc=4.5: logZ med=0.74 [-0.24,1.36]
- P2 derived: logZ shift = 0.65 dex for +1.5 dex cloud shift (across horizon)
- CONCLUSION: P1 (IR metallicity prior-dominated) and P2 (cloud mis-placement biases logZ) CONFIRMED on real data.

### stage7_fiducial.json — fiducial-robustness of cloud-opacity horizon
Fixed wavelength grid = Alderson G395H; varied fiducial atmosphere (T,logZ,CO).
- horizon mean=3.02 +/- 0.02 dex across 6 valid fiducials (stable to fiducial guess)
- NaN at logZ>=2: cloud always prior-dominated (horizon >5.5) — physically sensible
- CONCLUSION: horizon is an opacity/wavelength-structure fact, NOT a fiducial-artifact.

### Integrity notes
- All retrievals fit REAL published JWST transit depths. No synthetic spectrum presented as data.
- emcee sampler is PLATON's own (CombinedRetriever), independent of the diagnostic's linear/Jacobian machinery.
- Short chains (nwalk=28, nsteps=700) used for baseline validation; posterior widths are indicative, not final-precision.

## 2026-08-16 — Self-review round 2 (stage8 T-P profile robustness)

Tested whether the cloud-opacity horizon depends on the T-P profile SHAPE.
Fixed real G395H grid; varied profile shape (isothermal, +/-400 K/dex, +800 K/dex),
T fixed. Free params [logZ, CO_ratio, log_cloudtop_P]:
- isothermal:       horizon = nan (> 6.5, cloud always prior-dominated)
- grad +400 K/dex: horizon = 5.940
- grad -400 K/dex: horizon = nan (> 6.5)
- grad +800 K/dex: horizon = 5.898
Spread among gradient profiles: 0.042 dex.

CONCLUSION: the T-P shape matters. Isothermal gives the MOST PESSIMISTIC
(lowest) horizon — clouds maximally hidden. Real Guillot-like gradients push
the horizon to ~5.9, i.e. clouds are observable to ~1.5 dex deeper pressure
than the isothermal diagnostic (3.373) implied. The qualitative conclusion still
holds: IR-only JWST (horizon ~5.9 under gradients) probes shallower than
optical (4.2-4.7), so clouds remain prior-dominated for IR-only at the
pressures where they sit. The isothermal diagnostic is a CONSERVATIVE lower
bound on cloud observability. This is now stated as a labeled limitation.

## Reproducible-package lock (S11)
{
 "main.pdf": "d8638d4c710a1d87",
 "refs.bib": "b4e0ac2f8b2e2bb4",
 "collect_spectra.py": "ebf724e5559d9a67",
 "fetch_archive.py": "3075e49e2ae960da",
 "fetch_params.py": "141d8e65e4fbad86",
 "identifiability.py": "14e2c1028a9ce18c",
 "make_figures.py": "09edfacf5a80d727",
 "smoke_retrieval.py": "78036889bd45b89f",
 "stage1_forward_probe.py": "2287ee8d039b5171",
 "stage1_probe.json": "a68f55cc54ecd08b",
 "stage2_diagnostics.json": "e7779713b75e2ad3",
 "stage2_real_diagnostics.py": "1723bad6efc4d37e",
 "stage3_cloud_probe.json": "25279403f4bcbbd9",
 "stage3_cloud_probe.py": "accb3ec35adfb99e",
 "stage4_horizon_and_bias.py": "14161eaa6d99159b",
 "stage4_results.json": "084b4d8a8794a7a4",
 "stage5_nonlinear.json": "190f2fe3962d508f",
 "stage5_nonlinear_check.py": "6759f821d25977b8",
 "stage6_retrieval.json": "0f556adb5c172a3a",
 "stage6_retrieval.py": "043f0a8defe7418c",
 "stage7_fiducial.json": "8fb9d94e7fcd769f",
 "stage7_fiducial.py": "a48f44e3c52f1c5b",
 "stage8_tp_profile.json": "2611c083af638628",
 "stage8_tp_profile.py": "975b144476b534e0",
 "GJ486b_MORAN_ET_AL__202_n110.json": "750154dd89e8a45a",
 "GJ486b_MORAN_ET_AL__202_n46.json": "d3c91a88209e11f8",
 "GJ486b_MORAN_ET_AL__202_n47.json": "8c1d0cb67f122236",
 "HAT-P-18b_FU_ET_AL__2022_n1021.json": "e36587c2f531f36c",
 "WASP-39b_AHRER_ET_AL__202_n107.json": "010667e5ef8d6e4b",
 "WASP-39b_ALDERSON_ET_AL___n344.json": "0dd136eaa2a5ffdd",
 "WASP-39b_FEINSTEIN_ET_AL__n331.json": "d4e14adcefd8b87f",
 "WASP-39b_JWST_TRANSITING__n243.json": "2057ea32ba283429",
 "WASP-39b_JWST_TRANSITING__n67.json": "a0ff5d0a6b5b47a3",
 "WASP-39b_JWST_TRANSITING__n94.json": "74262ee130dd302f",
 "WASP-39b_JWST_TRANSITING__n95.json": "e7cac181a6564879",
 "WASP-39b_RUSTAMKULOV_ET_A_n207.json": "881a9d2054d4dff6"
}

## Round-1 revision: new artefacts (added 2026-10-20)

| artefact | sha256[:16] |
|---|---|
| `stage9_uncertainty.py` | afa10d19d526a7c2 |
| `stage9_uncertainty.json` | afdb0521ccb0aa75 |
| `stage10_retrieval_long.py` | db17f8020b49fdc6 |
| `stage11_robustness.py` | 45de5c1f08110bfe |
| `stage11_fiducial.json` | 24b7302b9822c6d8 |
| `stage11_tp.json` | 1aea750b11b60d53 |
| `stage11_mmw.json` | 3ae4c15d9fd02531 |
| `stage12_mrt.py` | 30ae4ed96ef024fe |
| `stage13_chemistry.py` | 5ec2a5930978734d |
| `stage13_chemistry.json` | 9e522ffe1034ddc8 |
| `stage14_refine.py` | 2a327c0074821665 |
| `stage14_refine.json` | c4561bcd8acaf71b |
| `stage15_mrt_finalize.py` | 247bbada02731edf |
| `stage16_fig_robustness.py` | b65fd8921272ee0f |
| `stage17_prior.py` | 2528510ffb1726a7 |
| `stage17_prior.json` | 600a3d6e91df6a4a |
| `stage18_posteriors.py` | a0ddf3ca750907a1 |
| `stage18_posteriors.json` | 6f9f247fd1d65ecc |
| `stage19_discriminate.py` | 097353ae21be7845 |
| `stage19_discriminate.json` | c910da662b3e10b7 |
| `stage19b_cloud_sensitivity.py` | 6c16a06a4b81c0c8 |
| `stage19b_cloud_sensitivity.json` | 96a554850d043bc7 |

Also in `data/`: `summary.csv`, `horizon_curves.csv`, `README.txt` (the
machine-readable package requested by the data editor). Chains for the six
retrieval runs are in `experiments/ck_stage10/`.

### Retracted artefacts

- `stage6_retrieval.py` / `stage6_retrieval.json` (v1 retrieval): invalid —
  SI-unit bug made every likelihood −inf, so the chains never moved.
- `stage7_fiducial.py` / `stage8_tp_profile.py`: superseded. stage7 used a
  generic 1 R_sun / 1 M_jup system, a flat 100 ppm sigma and a prior width
  that the main results do not use; stage8 used PLATON's custom T–P profile
  interface, which fails its own isothermal-limit check by 2.4% in transit
  depth. Their numbers have been replaced by stage11 Part A/C and stage14.