# GenZ Working Environment (Clasificacion Binaria) — UDD Bootcamp Modulo 7

## Proyecto-n-7-Data-Science

Repositorio del proyecto final del Bootcamp **Ciencia de Datos e Inteligencia Artificial (UDD)**.  
Se construye un modelo de **clasificacion binaria** para predecir el ambiente de trabajo preferido por la Generacion Z, y se publica una **API REST** (Flask + ngrok) para consumir predicciones via internet.

---

## 1) Objetivo
Construir un modelo que prediga el ambiente de trabajo preferido:
- `FULL_OFFICE`
- `REMOTE`

Dataset: **Understanding career aspirations of GenZ** (KultureHire, Kaggle).  
En este repo se incluye una copia del CSV para reproducibilidad.

---

## 2) Contexto: iteracion multiclase (3 clases) y decision final
Inicialmente se probo un enfoque **multiclase (3 clases)** agrupando las 6 categorias originales en:
- `OFFICE`
- `REMOTE`
- `HYBRID`

A pesar de iteraciones con **tuning** y **ensambles**, el desempeno multiclase fue limitado para los objetivos del proyecto (dataset pequeno y clases con patrones similares).  
Por esta razon se adopto una formulacion **binaria**, que permite un mejor control del balance de clases y una solucion mas estable para exponer via API.

---

## 3) Definicion del target binario
Target original (6 categorias) en la columna:

`What is the most preferred working environment for you.`

Se define:

- `FULL_OFFICE` = `"Every Day Office Environment"`
- `REMOTE` = resto de categorias (Fully Remote + Hybrid)

---

## 4) Dataset y variables de entrada (features)
El modelo usa **12 features** (11 categoricas + 1 numerica).  
Estas mismas 12 variables son el **input del endpoint** `/predict`:

1. `Your_Current_Country`
2. `Your_Gender`
3. `Factors_influencing_career_aspirations`
4. `Higher_Education_outside_India_self_sponsor`
5. `Likely_work_for_one_employer_3_years`
6. `Work_for_company_mission_not_defined`
7. `Work_for_company_mission_misaligned`
8. `Employers_you_would_work_with`
9. `Learning_environment`
10. `Manager_type`
11. `Preferred_setup`
12. `Likely_work_for_company_no_social_impact` (numerica 0 a 10)

Notas de preprocesamiento:
- Categorizacion de valores raros: categorias con baja frecuencia se reemplazan por `OTHER`.
- One-hot encoding para variables categoricas.
- El artefacto `joblib` incluye el pipeline completo (preprocesamiento + modelo).

---

## 5) Modelo final
- Algoritmo: **Logistic Regression (tuned)**
- Threshold decision: **0.50**
- Model version: **logreg_l2_c0p01_v1**
- Artefacto: `model_logreg_fulloffice_remote_v1.joblib`

---

## 6) Metricas (Test)

### 6.1 Reporte por clase
| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| FULL_OFFICE | 0.3750 | 0.6000 | 0.4615 | 10 |
| REMOTE | 0.8710 | 0.7297 | 0.7941 | 37 |

### 6.2 Metricas globales
| Metrica | Valor |
|---|---:|
| Accuracy | 0.7021 |
| F1 Macro | 0.6278 |
| F1 Weighted | 0.7234 |

Interpretacion breve:
- El modelo funciona mejor en `REMOTE` (clase mayoritaria).
- `FULL_OFFICE` es mas desafiante por menor soporte (menos ejemplos) y desbalance de clases.

---

## 7) API REST (Flask + ngrok)

### 7.1 Base URL (ngrok)
`https://nonbeneficial-uninverted-roseanna.ngrok-free.dev`

> Nota: el link de ngrok es temporal y puede cambiar si se reinicia el runtime de Google Colab o se vuelve a levantar el tunel.

### 7.2 Endpoints
- `GET /` -> health check
- `POST /predict` -> prediccion + confianza

### 7.3 Health check

**Request:**
```bash
curl -X GET "https://nonbeneficial-uninverted-roseanna.ngrok-free.dev/"
