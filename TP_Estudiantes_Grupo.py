import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 0. LISTA

set_de_datos = [
    {"edad": 54, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 7.44, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 66.32, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 51, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 7.0, "CantMateriasAprobadasPrimerCuatrimestre": 7, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 50.64, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 59, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 8.04, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 52.28, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 29, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 8.41, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 41.76, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 57, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 4.04, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 72.07, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 45, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 4.46, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 62.0, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 2, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 37, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 7.88, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 46.96, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 2, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 48, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 8.03, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 64.67, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 50, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 9.19, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 1, "AsistenciaPromedio(%)": 46.41, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 5, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 37, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 7.97, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 99.01, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 3, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 45, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 8.2, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 96.94, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 43, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 4.17, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 44.37, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 13, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 21, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 8.35, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 73.69, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 28, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 4.48, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 59.36, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 46, "genero": "m", "carrera": "TÉCNICATURA SUPERIOR EN HIGIENE Y SEGURIDAD EN EL TRABAJO", "PromedioPrimerCuatrimestre": 6.62, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 97.84, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 12, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 40, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 8.51, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 94.19, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 22, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 8.66, "CantMateriasAprobadasPrimerCuatrimestre": 7, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 51.97, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 60, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 8.43, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 79.54, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 58, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 4.34, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 69.93, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 45, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 6.53, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 91.59, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 23, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 8.6, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 1, "AsistenciaPromedio(%)": 53.24, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 11, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 32, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 5.35, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 64.05, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 45, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 6.03, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 92.77, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 3, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 20, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 6.84, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 46.11, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 16, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 24, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 7.14, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 44.36, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 6, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 29, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 8.6, "CantMateriasAprobadasPrimerCuatrimestre": 7, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 91.77, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 38, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 7.86, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 73.26, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 6, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 35, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 4.25, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 91.7, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 47, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 7.89, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 53.88, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 18, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 4.22, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 98.65, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 39, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 7.83, "CantMateriasAprobadasPrimerCuatrimestre": 7, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 58.95, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 50, "genero": "m", "carrera": "TÉCNICATURA SUPERIOR EN HIGIENE Y SEGURIDAD EN EL TRABAJO", "PromedioPrimerCuatrimestre": 8.39, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 42.24, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 36, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 6.41, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 59.13, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 7, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 31, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 4.34, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 48.73, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 49, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 5.06, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 70.57, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 17, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 56, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 9.62, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 54.41, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 21, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 4.25, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 78.37, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 16, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 40, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 6.5, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 42.55, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 1, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 27, "genero": "f", "carrera": "TÉCNICATURA SUPERIOR EN HIGIENE Y SEGURIDAD EN EL TRABAJO", "PromedioPrimerCuatrimestre": 9.06, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 76.59, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 2, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 24, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 8.96, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 90.71, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 38, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 4.19, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 84.3, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 2, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 33, "genero": "m", "carrera": "TÉCNICATURA SUPERIOR EN HIGIENE Y SEGURIDAD EN EL TRABAJO", "PromedioPrimerCuatrimestre": 7.95, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 72.67, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 19, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 42, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 8.23, "CantMateriasAprobadasPrimerCuatrimestre": 7, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 69.81, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 20, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 39, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 8.1, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 1, "AsistenciaPromedio(%)": 47.57, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 26, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 4.46, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 90.95, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 12, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 20, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 9.91, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 69.73, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 7, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 54, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 5.17, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 67.24, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 13, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 21, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 5.2, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 74.71, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 55, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 6.37, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 52.28, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 25, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 5.34, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 97.8, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 1, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 53, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 9.99, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 55.59, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 25, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 7.14, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 91.35, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 43, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 4.61, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 68.36, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 17, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 28, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 7.69, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 61.5, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 6, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 51, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 8.7, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 48.09, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 56, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 6.12, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 61.71, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 19, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 39, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 6.63, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 52.24, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 11, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 55, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 8.19, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 53.99, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 27, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 8.35, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 69.0, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 10, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 26, "genero": "f", "carrera": "TÉCNICATURA SUPERIOR EN HIGIENE Y SEGURIDAD EN EL TRABAJO", "PromedioPrimerCuatrimestre": 8.35, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 96.98, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 17, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 42, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 7.94, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 92.58, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 40, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 9.24, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 50.82, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 5, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 55, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 6.75, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 97.1, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 19, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 8.29, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 59.23, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 60, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 4.55, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 54.4, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 12, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 60, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 9.21, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 93.74, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 52, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 7.77, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 86.73, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 7, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 42, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 9.46, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 61.46, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 3, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 59, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 5.74, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 79.48, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 6, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 42, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 9.73, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 46.89, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 47, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 9.6, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 91.31, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 10, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 23, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 7.84, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 43.98, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 3, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 34, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 9.74, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 58.79, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 5, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 32, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 9.4, "CantMateriasAprobadasPrimerCuatrimestre": 7, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 87.72, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 10, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 50, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 6.21, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 69.3, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 10, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 53, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 5.79, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 62.91, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 27, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 9.74, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 75.67, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 50, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 5.52, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 95.89, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 3, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 40, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 4.88, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 91.09, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 13, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 50, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 6.28, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 51.05, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 13, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 50, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 7.46, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 83.91, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 20, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 19, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 8.12, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 69.59, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 40, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL", "PromedioPrimerCuatrimestre": 5.59, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 2, "AsistenciaPromedio(%)": 72.6, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 15, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 22, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 6.53, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 88.34, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 19, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 7.06, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 50.64, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 7, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 51, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 8.65, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 46.04, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 12, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 43, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ANÁLISIS DE SISTEMAS", "PromedioPrimerCuatrimestre": 5.18, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 91.19, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 10, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 45, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 5.91, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 42.22, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 7, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 44, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 4.32, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 71.57, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 20, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 56, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 9.45, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 80.4, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 17, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 39, "genero": "m", "carrera": "TÉCNICATURA SUPERIOR EN HIGIENE Y SEGURIDAD EN EL TRABAJO", "PromedioPrimerCuatrimestre": 4.06, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 47.89, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 5, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 39, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 7.17, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 81.77, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 3, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 29, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ENFERMERÍA", "PromedioPrimerCuatrimestre": 6.67, "CantMateriasAprobadasPrimerCuatrimestre": 5, "CantMateriasDesaprobadasPrimerCuatrimestre": 7, "AsistenciaPromedio(%)": 98.74, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 8, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Continúa"},
    {"edad": 42, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN DE PYMES", "PromedioPrimerCuatrimestre": 7.01, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 3, "AsistenciaPromedio(%)": 87.44, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 12, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 22, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN COMUNICACIÓN MULTIMEDIAL", "PromedioPrimerCuatrimestre": 7.42, "CantMateriasAprobadasPrimerCuatrimestre": 3, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 48.8, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 16, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
    {"edad": 50, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN RADIOLOGIA", "PromedioPrimerCuatrimestre": 6.04, "CantMateriasAprobadasPrimerCuatrimestre": 4, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 85.32, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 9, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Continúa"},
    {"edad": 47, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 7.33, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 6, "AsistenciaPromedio(%)": 47.77, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 19, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 40, "genero": "f", "carrera": "TECNICATURA SUPERIOR EN DISEÑO INDUSTRIAL", "PromedioPrimerCuatrimestre": 7.05, "CantMateriasAprobadasPrimerCuatrimestre": 6, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 64.68, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 18, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 21, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 9.93, "CantMateriasAprobadasPrimerCuatrimestre": 2, "CantMateriasDesaprobadasPrimerCuatrimestre": 5, "AsistenciaPromedio(%)": 57.96, "trabaja/NoTrabaja": "No", "DistanciaDomicilioAlInstituto(Kms)": 14, "ActividadesExtracurriculares(Estudio)": "Si", "EstadoFinal": "Abandonó"},
    {"edad": 50, "genero": "m", "carrera": "TECNICATURA SUPERIOR EN ADMINISTRACIÓN CONTABLE", "PromedioPrimerCuatrimestre": 7.46, "CantMateriasAprobadasPrimerCuatrimestre": 1, "CantMateriasDesaprobadasPrimerCuatrimestre": 4, "AsistenciaPromedio(%)": 54.12, "trabaja/NoTrabaja": "Si", "DistanciaDomicilioAlInstituto(Kms)": 4, "ActividadesExtracurriculares(Estudio)": "No", "EstadoFinal": "Abandonó"},
]

# ________________________________________________________________________________________________________
# 1. ANÁLISIS EXPLORATORIO DE DATOS Y CÁLCULO DE ENTROPÍA H(S)
# ________________________________________________________________________________________________________

print("\n" + "_" * 60)
print("ANÁLISIS EXPLORATORIO DE DATOS:")
print("_" * 60)

estudiantes_abandonaron = [d for d in set_de_datos if d["EstadoFinal"] == "Abandonó"]
estudiantes_continuan = [d for d in set_de_datos if d["EstadoFinal"] == "Continúa"]

total_abandonaron = len(estudiantes_abandonaron)
print("-Estudiantes que abandonaron:", total_abandonaron)

total_continuan = len(estudiantes_continuan)
print("-Estudiantes que continúan:", total_continuan)

filtro_edad_abandonaron = [d["edad"] for d in estudiantes_abandonaron]
print("-Edades de estudiantes que abandonaron:", filtro_edad_abandonaron)

filtro_edad_continuan = [d["edad"] for d in estudiantes_continuan]
print("-Edades de estudiantes que continúan:", filtro_edad_continuan)

filtro_asistencia_abandonaron = [d["AsistenciaPromedio(%)"] for d in estudiantes_abandonaron]
print("-Asistencia promedio de estudiantes que abandonaron:", filtro_asistencia_abandonaron,)

filtro_asistencia_continuan = [d["AsistenciaPromedio(%)"] for d in estudiantes_continuan]
print("-Asistencia promedio de estudiantes que continúan:", filtro_asistencia_continuan,)

trabajan_abandonaron = sum(1 for d in estudiantes_abandonaron if d["trabaja/NoTrabaja"] == "Si")
print("-Estudiantes que trabajan y abandonaron:", trabajan_abandonaron)

no_trabajan_abandonaron = sum(1 for d in estudiantes_abandonaron if d["trabaja/NoTrabaja"] == "No")
print("-Estudiantes que no trabajan y abandonaron:", no_trabajan_abandonaron)

#ENTROPÍA:

def calcular_entropia(lista_estudiantes):
  total_alumnos = len(lista_estudiantes)
  if total_alumnos == 0:
    return 0.0

  alumnos_abandonaron = sum(1 for d in lista_estudiantes if d["EstadoFinal"] == "Abandonó")
  alumnos_continuan = total_alumnos - alumnos_abandonaron

  if alumnos_abandonaron == 0 or alumnos_continuan == 0:
    return 0.0

  probabilidad_abandono = alumnos_abandonaron / total_alumnos
  probabilidad_continua = alumnos_continuan / total_alumnos
  return -(probabilidad_abandono * math.log2(probabilidad_abandono) + probabilidad_continua * math.log2(probabilidad_continua))

entropia_inicial = calcular_entropia(set_de_datos)
print(f"\n-Entropía Inicial del Conjunto H(S): {entropia_inicial:.4f}\n")

# ________________________________________________________________________________________________________
# 2. PREPARACIÓN DE DATOS
# ________________________________________________________________________________________________________

datos_estudiantes = pd.DataFrame(set_de_datos)

variables_entrada = pd.get_dummies(datos_estudiantes.drop(columns=["EstadoFinal"]), drop_first=True)
variable_objetivo = datos_estudiantes["EstadoFinal"]

(
    variables_entrenamiento,
    variables_prueba,
    objetivo_entrenamiento,
    objetivo_prueba,
) = train_test_split(
    variables_entrada,
    variable_objetivo,
    test_size=0.2,
    random_state=42,
    stratify=variable_objetivo,
)

# ________________________________________________________________________________________________________
# 3. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN
# ________________________________________________________________________________________________________

modelo_arbol = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
modelo_arbol.fit(variables_entrenamiento, objetivo_entrenamiento)

# ________________________________________________________________________________________________________
# 4. EVALUACIÓN DEL MODELO CON CONJUNTO DE PRUEBA
# ________________________________________________________________________________________________________

predicciones_prueba = modelo_arbol.predict(variables_prueba)

print(
    "Exactitud (Accuracy):"
    f" {accuracy_score(objetivo_prueba, predicciones_prueba):.2%}"
)
print("\n-Matriz de Confusión:")
print(confusion_matrix(objetivo_prueba, predicciones_prueba))

# REPORTE QUE PASO A ESPAÑOL
reporte_diccionario = classification_report(
    objetivo_prueba, predicciones_prueba, output_dict=True
)
reporte_espaniol = pd.DataFrame(reporte_diccionario).transpose().rename(
    columns={
        "precision": "Precisión",
        "recall": "Sensibilidad",
        "f1-score": "Puntuación F1",
        "support": "Soporte",
    },
    index={
        "accuracy": "Exactitud",
        "macro avg": "Promedio Macro",
        "weighted avg": "Promedio Ponderado",
    },
).round(2)

print("\n-Reporte de Clasificación:")
print(reporte_espaniol)

# ________________________________________________________________________________________________________
# 5. FUNCIÓN DE PREDICCIÓN PARA NUEVOS ESTUDIANTES INGRESADOS
# ________________________________________________________________________________________________________


def predecir_nuevo_estudiante(lista_nuevos_ingresos):
  print("\n" + "_" * 60)
  print("DATOS DEL NUEVO ESTUDIANTE INGRESADO:")
  print("_" * 60)

  for i, estudiante in enumerate(lista_nuevos_ingresos, start=1):
    print(f"\n--- Estudiante #{i} ---")
    for clave, valor in estudiante.items():
      print(f"- {clave}: {valor}")

  datos_nuevos = pd.DataFrame(lista_nuevos_ingresos)
  datos_nuevos_codificados = pd.get_dummies(
      datos_nuevos, drop_first=True
  ).reindex(columns=variables_entrada.columns, fill_value=0)

  predicciones = modelo_arbol.predict(datos_nuevos_codificados)
  probabilidades = modelo_arbol.predict_proba(datos_nuevos_codificados)

  clases_modelo = list(modelo_arbol.classes_)
  indice_abandono = clases_modelo.index("Abandonó")

  print("\n" + "_" * 60)
  print("RESULTADO DE LA PREDICCIÓN:")
  print("_" * 60)
  for i, resultado_prediccion in enumerate(predicciones, start=1):
    porcentaje_probabilidad = probabilidades[i - 1][indice_abandono] * 100
    print(
        f"Estudiante #{i}: Predicción = {resultado_prediccion} | Probabilidad de"
        f" Abandono = {porcentaje_probabilidad:.2f}%\n"
    )


# ________________________________________________________________________________________________________
# 6. PRUEBA CON UN NUEVO ESTUDIANTE INGRESANTE
# ________________________________________________________________________________________________________

nuevo_estudiante = [{
    "edad": 20,
    "genero": "f",
    "carrera": (
        "TECNICATURA SUPERIOR EN CIENCIA DE DATOS E INTELIGENCIA ARTIFICIAL"
    ),
    "PromedioPrimerCuatrimestre": 8.5,
    "CantMateriasAprobadasPrimerCuatrimestre": 5,
    "CantMateriasDesaprobadasPrimerCuatrimestre": 0,
    "AsistenciaPromedio(%)": 92.0,
    "trabaja/NoTrabaja": "No",
    "DistanciaDomicilioAlInstituto(Kms)": 4,
    "ActividadesExtracurriculares(Estudio)": "Si",
}]

# Ejecución de la prueba
predecir_nuevo_estudiante(nuevo_estudiante)