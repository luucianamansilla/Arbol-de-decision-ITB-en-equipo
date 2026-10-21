# Predicción de Abandono en Estudiantes Terciarios 🎓📊

**Instituto Tecnológico Beltrán (ITB) — Modelo de Aprendizaje Automático con Árboles de Decisión**

Este proyecto implementa un modelo de Machine Learning supervisado mediante el algoritmo de Árbol de Decisión (`DecisionTreeClassifier`) basado en el criterio de Entropía. Su objetivo principal es predecir de manera temprana la probabilidad de deserción de estudiantes de carreras terciarias en base a variables sociodemográficas, académicas e institucionales recabadas durante su primer año de cursada.

---

## 🛠️ Requisitos del Sistema

Para ejecutar este script necesitás tener instalado **Python 3.8+** junto a las siguientes librerías:

- **pandas** — Manipulación de estructuras de datos y DataFrames.
- **scikit-learn** — Entrenamiento, división y evaluación del modelo de Machine Learning.
- **math** — Módulo nativo de Python para cálculos matemáticos logarítmicos.

### Instalación de Dependencias

Podés instalar las dependencias ejecutando en la terminal:

```bash
pip install pandas scikit-learn
```

(O `python -m pip install pandas scikit-learn` en Windows).

---

## 📁 Estructura e Implementación del Código

El script principal (`TP_Estudiantes_Grupo.py`) está organizado secuencialmente en las siguientes secciones:

1. **Análisis Exploratorio de Datos (EDA) & Entropía Inicial H(S):**
   - Filtrado y conteo exploratorio de variables cuantitativas y cualitativas.
   - Cálculo programático explícito de la Entropía Inicial del conjunto de datos mediante logaritmos en base 2:

     `H(S) = -p_a * log2(p_a) - p_c * log2(p_c)`

2. **Preprocesamiento y Codificación de Datos:**
   - Conversión de la estructura primaria a un DataFrame de Pandas.
   - Codificación de variables categóricas nominales (One-Hot Encoding) mediante `pd.get_dummies()`.

3. **División del Conjunto de Datos:**
   - Separación en Entrenamiento (80%) y Prueba (20%) con conservación de proporciones estratificadas (`stratify=y`).

4. **Entrenamiento del Modelo:**
   - Construcción del clasificador `DecisionTreeClassifier` configurado con `criterion="entropy"` y una profundidad máxima acotada (`max_depth=4`) para prevenir el sobreajuste (*overfitting*).

5. **Evaluación de Desempeño:**
   - Cálculo de la métrica general de acierto (*Accuracy*).
   - Generación de la Matriz de Confusión (Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Verdaderos Negativos).
   - Reporte completo de clasificación parametrizado al español (Precisión, Sensibilidad/Recall, Puntuación F1 y Soporte).

6. **Función de Predicción para Nuevos Ingresantes:**
   - Módulo ejecutable `predecir_nuevo_estudiante()` que recibe una estructura con datos de un nuevo alumno, imprime en consola los valores ingresados y calcula la probabilidad porcentual de abandono/continuidad.

---

## 📋 Desarrollo de las Tareas de la Actividad

### 1. Analizar y preparar los datos

El dataset histórico cuenta con **100 registros** iniciales. Presenta un balance con **54 estudiantes** en condición de *"Abandonó"* y **46** en condición de *"Continúa"*, lo que genera una Entropía Inicial del sistema **H(S) = 0.9954** (cercana a la máxima incertidumbre).

### 2. Entrenar un árbol de decisión utilizando un conjunto de entrenamiento

Se entrenó el modelo utilizando el 80% de la muestra (80 casos) utilizando el algoritmo de entropía para seleccionar los nodos de decisión con mayor ganancia de información.

### 3. Evaluar el modelo con un conjunto de prueba

Sobre el 20% de los datos reservados para evaluación (20 casos de prueba no vistos por el modelo), se obtuvieron los siguientes resultados:

| Métrica | Resultado |
|---|---|
| Exactitud (Accuracy) | 35.00% |
| Matriz de Confusión | `[[6, 5], [8, 1]]` |
| Precisión para clase "Abandonó" | 0.43 (43%) |
| Sensibilidad / Recall para clase "Abandonó" | 0.55 (55%) |

### 4. Interpretar el árbol generado: ¿Qué variables parecen ser más importantes para predecir el abandono?

De acuerdo a las divisiones realizadas en el árbol de decisión y los cálculos de ganancia de información, las variables con mayor peso predictivo son:

- **Cantidad de materias desaprobadas en el primer cuatrimestre:** un número elevado de materias reprobadas actúa como el nodo raíz principal para predecir la deserción.
- **Promedio del primer cuatrimestre:** promedios generales inferiores a 6.00 incrementan drásticamente la probabilidad de abandono.
- **Edad del estudiante:** muestra patrones diferenciales entre estudiantes recién egresados del secundario y adultos que equilibran responsabilidades laborales/familiares.
- **Asistencia promedio (%):** niveles de asistencia discontinuos anteceden al abandono definitivo.

### 5. Proponer recomendaciones para la institución basadas en los resultados del modelo

- **Implementación de un Sistema de Alertas Tempranas (SAT):** configurar una alerta automática al finalizar el primer cuatrimestre para identificar alumnos con más de 2 materias reprobadas o promedios menores a 6.0.
- **Refuerzo de Tutorías Académicas:** orientar programas de acompañamiento específico a las asignaturas del primer cuatrimestre que registran mayores índices de reprobación.
- **Monitoreo de Asistencia y Acompañamiento Etario:** establecer contacto preventivo cuando la asistencia disminuya e implementar modalidades flexibles de estudio dirigidas a los rangos etarios más vulnerables.

---

## 🚀 Ejecución del Proyecto

Para correr el código en tu equipo:

```bash
python TP_Estudiantes_Grupo.py
```
