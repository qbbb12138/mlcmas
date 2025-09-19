# mlcmas
Data-driven prediction of RESs' anti-CMAS corrosion performance.

**Environment**
- Python = 3.9

---

1_Compound_Generator  
One-liner: Systematically enumerates candidate compositions in the order **Element Pool → Sampling Density → Formula Constituents**.  
|

2_Parameter_Compound  
Function: Assigns a unique descriptor ID to each formula. Si and O are used **only in the normalization denominator**, enabling phase discrimination between **monosilicate (RE₂SiO₅)** and **disilicate (RE₂Si₂O₇)**.  
|

|(Optional) 5_Betaphase_prediction  
Function: Implements an improved **Voting ensemble model** to classify and predict the **β-RE₂Si₂O₇ phase**.  
|

3_Corrosion_Model  
**XGBoost Baseline & 5-Fold KFold Models (for CRG Prediction)**  
- Goal: Two reproducible pipelines to compare and predict **CRG**.  
- Scripts:  
  - **XGBoost Baseline**: `xgb_predit.py` (aka `xgb_predict.py`)  
  - **5-Fold KFold**: `corrosion_model.py`  
|

4_Visualization_Spatial  
Visualize how **CRG** varies across the feature space with **3D trends** and **2D slice localization**.  
- **3D Trend Map**: Run `plot_heatmap_3d.m` to generate the overall CRG distribution across (X, Y, Z).  
- **Filter by Range**: Run `filter_xyz.m` to crop the dataset within user-defined coordinate ranges.  
- **2D Slice Map**: Run `plot_heatmap_2d.m` to plot CRG heatmaps on the (X–Y) plane while fixing one axis.  
