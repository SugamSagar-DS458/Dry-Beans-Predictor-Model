# 🌱 Dry Beans Classification & Interactive Predictor

An end-to-end Machine Learning project that classifies seven different registered varieties of dry beans based on computer vision-derived morphological, shape, and geometric features. The repository includes an exploratory data analysis (EDA) and model experimentation pipeline alongside an interactive **Streamlit** web application for real-time predictions.

---

## 📌 Project Overview

Bean seed certification and classification traditionally require significant manual labor and specialized expertise. This project leverages standard geometric and structural features—such as area, perimeter, axes lengths, eccentricity, and form factors—extracted from high-resolution grain imagery to automate variety classification.

### Target Classes (7 Bean Varieties)
1. **Seker**
2. **Barbunya**
3. **Bombay**
4. **Cali**
5. **Horoz**
6. **Sira**
7. **Dermason**

---

## 🗂️ Project Structure

```text
├── Dry_Beans_Classifications.ipynb   # Model development & exploratory data analysis
├── Dry_Beans_Predictor.py            # Streamlit web application for interactive inference
├── Dry_Beans_Dataset.xlsx            # Dataset containing morphological features
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 📊 Features & Dataset

The dataset contains morphological features extracted from grain images:

| Feature Name | Description |
| :--- | :--- |
| `Area` | The area of a bean zone and the number of pixels within its boundaries. |
| `Perimeter` | Bean circumference defined as the length of its border. |
| `MajorAxisLength` | Distance between the ends of the longest line drawn through the bean. |
| `MinorAxisLength` | Longest line that can be drawn orthogonal to the major axis. |
| `AspectRation` | Relationship between Major Axis Length and Minor Axis Length ($L_{major} / L_{minor}$). |
| `Eccentricity` | Eccentricity of the ellipse having the same moments as the region. |
| `ConvexArea` | Number of pixels in the smallest convex polygon enclosing the bean. |
| `EquivDiameter` | Diameter of a circle with the same area as the bean ($\sqrt{4 \times \text{Area} / \pi}$). |
| `Extent` | Ratio of the pixels in the bounding box to the bean area. |
| `Solidity` | Ratio of the pixels in the convex hull to those in the bean (`Area / ConvexArea`). |
| `roundness` | Roundness metric calculated as $(4 \times \pi \times \text{Area}) / \text{Perimeter}^2$. |
| `Compactness` | Measures the roundness of the object (`EquivDiameter / MajorAxisLength`). |
| `ShapeFactor1 - 4` | Various dimensionless shape descriptors defining contour irregularities. |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/dry-beans-classification.git
cd dry-beans-classification
```

### 2. Create a Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Jupyter Notebook
To inspect exploratory data analysis, visualizations, and comparative model evaluation:
```bash
jupyter notebook Dry_Beans_Classifications.ipynb
```

### Streamlit Web Application
To launch the interactive predictor dashboard locally:
```bash
streamlit run Dry_Beans_Predictor.py
```
Open your browser at `http://localhost:8501` to test custom bean measurements and view prediction probabilities.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning & Preprocessing:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy, OpenPyXL
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Web Deployment:** Streamlit

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and adapt it for educational and practical applications.