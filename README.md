# Random Walk Simulation

> 機率模型及數據科學 2026 — Random Walk 作業

本專案實作 D 維隨機行走（Random Walk）模擬，涵蓋 D = 1, 2, 3, 4 維度，並根據模擬結果生成統計分析圖表與 LaTeX 報告。

## 📁 專案結構

```
Randomwalk/
├── main.cpp          # C++ 模擬主程式（OpenMP 加速）
├── Makefile          # 編譯、執行、繪圖、報告一鍵完成
├── figure.py         # Python 繪圖腳本（matplotlib）
├── main.tex          # LaTeX 報告原始碼
├── README.md         # 本文件
├── data/             # 模擬輸出 CSV 資料（自動產生）
└── figures/          # 圖表輸出（自動產生）
```

## 🔧 環境需求

- **C++ 編譯器**：GCC 15+（Homebrew `g++-15`），需支援 OpenMP
- **Python 3**：搭配 `numpy`、`matplotlib`
- **LaTeX**：`pdflatex`（如需編譯報告，可用 `brew install --cask basictex` 安裝）

## 🚀 快速開始

### 1. 編譯模擬程式

```bash
make build
```

### 2. 執行全部模擬

```bash
make run
```

這會對所有 D ∈ {1,2,3,4} × n ∈ {10², 10³, 10⁴, 10⁵, 10⁶} 執行 1000 次隨機行走，結果寫入 `data/` 目錄。

### 3. 生成圖表

```bash
make plot
```

圖表輸出至 `figures/` 目錄（PDF + PNG 格式）。

### 4. 編譯 LaTeX 報告

```bash
make report
```

### 一鍵完成

```bash
make all    # build → run → plot
```

## ⚙️ 命令列參數

模擬程式支援以下參數：

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `-d` | 維度 | 1, 2, 3, 4（全部） |
| `-n` | 步數 | 100 ~ 1,000,000 |
| `-w` | 每組隨機行走次數 | 1000 |
| `-s` | 隨機種子 | 系統時間 |
| `-o` | 輸出目錄 | `data` |

範例：

```bash
./rw_sim -d 2 -n 10000 -w 500 -s 42 -o my_data
```

## 📊 輸出資料

模擬會為每個 (D, n) 組合產生四類 CSV：

| 檔案 | 內容 |
|------|------|
| `dist_D{D}_n{n}.csv` | 最終位置、L1/L2 距離 |
| `section_D{D}_n{n}.csv` | 各象限停留步數 |
| `return_D{D}_n{n}.csv` | 回到原點的時間步 |
| `onedim_n{n}.csv` | 1D 專屬：n₋, n₀, n₊, m/n |

## 📈 圖表說明

| 圖表 | 說明 |
|------|------|
| `distance_n{n}.pdf` | 各維度 L1/L2 距離直方圖（4×2 grid，含平滑曲線與統計標註） |
| `section_D{D}.pdf` | 象限佔比驗證圖 |
| `return_to_origin.pdf` | 首次回到原點分佈 |
| `expected_return.pdf` | 平均首次回到原點步數 vs 行走長度 |
| `num_returns.pdf` | 回到原點次數分佈 |
| `m_over_n.pdf` | m/n 分佈（Arcsine Law） |
| `distance_scaling.pdf` | 距離與 √n 的 scaling 關係 |

## 📖 理論背景

- **Pólya's Recurrence Theorem**：1D 和 2D 隨機行走以機率 1 回到原點；3D 以上回到原點的機率 < 1
- **距離 Scaling**：經過 n 步後，期望距離 ∝ √n
- **Arcsine Law**：1D 隨機行走中，粒子在正半軸停留時間的比例服從 arcsine 分佈
