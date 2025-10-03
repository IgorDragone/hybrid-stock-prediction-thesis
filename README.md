# Modelo Predictivo Híbrido para la Inversión en Renta Variable

### Un sistema de recomendación basado en el análisis técnico, fundamental y macroeconómico

## 👨‍💻 Autor

Igor Dragone - alu0101469652@ull.edu.es \
Universidad de La Laguna \
Grado en Ingeniería Informática \
Curso 2025/2026

## 📌 Descripción del proyecto

Este proyecto forma parte de mi trabajo de fin de grado/tesis y se centra en el desarrollo de un modelo predictivo híbrido para la inversión en renta variable.
El modelo integra tres perspectivas:

* **Análisis Técnico** (series temporales, indicadores)
* **Análisis Fundamental** (estados financieros)
* **Análisis Macroeconómico** (masa monetaria y factores externos)

El objetivo es generar señales de compra/venta más precisas que los métodos convencionales, combinando la visión de corto y largo plazo.



## 🏗️ Estructura del repositorio

```
hybrid-stock-prediction-thesis/
│── README.md
│── requirements.txt
│── .gitignore
│── data/                 #datasets
│── notebooks/             # experimentos en Jupyter
│── src/                   # código modular
│── results/
```

---

<!-- / ## ⚙️ Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/IgorDragone/hybrid-stock-prediction-thesis.git
cd hybrid-stock-prediction-thesis
```

### 2. Crear el entorno virtual

Con **conda**:

```bash
conda create -n hybrid-stock python=3.10
conda activate hybrid-stock
pip install -r requirements.txt
```

Con **pip**:

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
```

---

## 📊 Fuentes de datos

* **Técnico**: Yahoo Finance, Alpha Vantage
* **Fundamental**: Financial Modeling Prep API, Edgar SEC
* **Macroeconómico**: Federal Reserve (FRED API)

⚠️ Los datasets en bruto no se suben a GitHub.
Para replicar: ver `src/data_loader.py` o `data_sources.md`.

---

## 🔬 Metodología

1. **Preprocesamiento & Feature Engineering**

   * Normalización, tratamiento de valores faltantes
   * Indicadores técnicos (RSI, MACD, etc.)
   * Variables fundamentales y macroeconómicas

2. **Modelado**

   * Algoritmos ML/DL (Random Forest, XGBoost, LSTM, etc.)
   * Modelo híbrido de fusión

3. **Evaluación**

   * Métricas: RMSE, Sharpe Ratio, precisión de señales buy/sell
   * Backtesting en dataset independiente

---

## 📈 Resultados esperados

* Mayor precisión que los modelos tradicionales
* Señales más robustas en escenarios de volatilidad
* Validación con datos reales y simulación de portafolio

---

## 🛠️ Tecnologías utilizadas 

* **Lenguajes:** Python (pandas, numpy, scikit-learn, tensorflow/pytorch)
* **Gestión de datos:** Jupyter, SQL, APIs financieras
* **Control de versiones:** Git & GitHub
* **Documentación:** LaTeX (Overleaf/GitHub sync) 

--- -->
