# Machine Learning Group Task - House Prices Prediction

A comprehensive machine learning project divided into 4 groups, each handling different stages of the ML pipeline using the House Prices dataset from Kaggle.

## 📁 Project Structure

```
house-prices-advanced-regression-techniques/
│
├── 01_Documentation/          📚 Guides and reports
│   ├── START_HERE.txt         ⭐ Read this first!
│   ├── ML_GROUP_TASK_README.md
│   ├── QUICK_START_GUIDE.md
│   └── PRESENTATION_OUTLINE.md
│
├── 02_Data/                   📊 Data files
│   ├── raw/                   (Original dataset)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── data_description.txt
│   └── processed/             (Cleaned data)
│
├── 03_Groups/                 👥 Group work
│   ├── Group_A/               (Define Objective & Collect Data)
│   ├── Group_B/               (Prepare Data & Select Algorithm)
│   ├── Group_C/               (Train & Test Models)
│   └── Group_D/               (Predict & Visualize Results)
│
├── 04_Outputs/                📈 Generated results
│   ├── visualizations/        (PNG charts)
│   ├── models/                (Trained models)
│   └── reports/               (Analysis reports)
│
├── 05_Logs/                   📝 Execution logs
│
├── 06_Config/                 ⚙️ Configuration
│
├── venv/                      🐍 Virtual environment
│
├── requirements.txt           📦 Python dependencies
├── .gitignore                 🚫 Git ignore rules
└── README.md                  📖 This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment (if not already done)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Read Documentation
Start with: `01_Documentation/START_HERE.txt`

### 3. Run Group Scripts (in order)
```bash
# Group A: Data Analysis
python 03_Groups/Group_A/data_analysis.py

# Group B: Data Preparation
python 03_Groups/Group_B/data_preparation.py

# Group C: Model Training
python 03_Groups/Group_C/model_training.py

# Group D: Visualization
python 03_Groups/Group_D/visualization_analysis.py
```

### 4. View Results
Check visualizations in: `04_Outputs/visualizations/`

## 📊 Dataset

- **Source**: Kaggle - House Prices: Advanced Regression Techniques
- **Training samples**: 1,460
- **Test samples**: 1,459
- **Features**: 79
- **Target**: SalePrice (in dollars)

## 👥 Groups Overview

| Group | Task | Files |
|-------|------|-------|
| **A** | Define Objective & Collect Data | data_analysis.py, GROUP_A_REPORT.md |
| **B** | Prepare Data & Select Algorithm | data_preparation.py, GROUP_B_REPORT.md |
| **C** | Train & Test Models | model_training.py, GROUP_C_REPORT.md |
| **D** | Predict & Visualize Results | visualization_analysis.py, GROUP_D_REPORT.md |

## 📦 Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

## 📚 Documentation Files

- **START_HERE.txt** - Quick orientation guide
- **ML_GROUP_TASK_README.md** - Comprehensive overview
- **QUICK_START_GUIDE.md** - Quick reference
- **PRESENTATION_OUTLINE.md** - Presentation templates

## ✅ Verification

All scripts have been tested and verified:
- ✓ Group A tested and working
- ✓ Visualizations generating correctly
- ✓ All file paths updated
- ✓ No broken references

## 🎯 Learning Outcomes

After completing this project, you will understand:
- ✅ All 8 stages of a real ML project
- ✅ How to find and prepare data
- ✅ How models are trained and evaluated
- ✅ How to visualize and interpret results
- ✅ How ML tools integrate in practice
- ✅ Real-world ML workflows

## 📝 Notes

- The project uses a supervised learning regression approach
- Data preprocessing includes handling missing values and feature encoding
- Models are evaluated using MSE, MAE, and R² metrics
- All visualizations are saved as PNG files in `04_Outputs/visualizations/`

## 🔧 Troubleshooting

If you encounter issues:
1. Ensure virtual environment is activated
2. Check that all dependencies are installed: `pip install -r requirements.txt`
3. Verify data files exist in `02_Data/raw/`
4. Check file paths in scripts match the directory structure

## 📞 Support

For detailed information:
- Read: `01_Documentation/START_HERE.txt`
- Check: `01_Documentation/QUICK_START_GUIDE.md`
- Review: `03_Groups/Group_X/GROUP_X_REPORT.md`

---

**Status**: ✅ Complete & Ready to Use  
**Created**: 2025-10-22  
**Structure**: Professional & Organized

