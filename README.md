⚙️ Metodología
1. Extracción de datos
Web scraping desde la Superintendencia de Pensiones de Chile

Limpieza y estructuración de datos por tipo de activo

2. Simulación
Procesos de Lévy para variables con colas gruesas y saltos (RV, FX, alternativos)
Vasicek o CIR para renta fija local
Correlación entre series vía copulas o matriz empírica

3. Optimización
Modelo de Generative Adversarial Networks (GANs) para generar y seleccionar las mejores trayectorias de ponderaciones 𝜆t

Objetivo: maximizar rendimiento ajustado por riesgo (Sharpe o Sortino) a través del tiempo

📈 Resultados esperados
Trayectoria promedio del portafolio simulado
Percentil 1% más negativo (escenario extremo)
Evolución mensual de las asignaciones óptimas
Visualización de sensibilidad por factor

🤝 ¿Colaboras?
Este es un proyecto experimental y abierto. Si tienes experiencia en:
Finanzas cuantitativas
Series de tiempo
Deep learning / GANs
Visualización o scraping financiero

...¡me encantaría que sumaras! Puedes abrir un issue, proponer ideas o escribirme directamente.
=======
# Simulaciones Vasicek con datos AFP

Este proyecto incluye un script (`vasicek_simulation.py`) que intenta descargar
los valores cuota del Fondo A desde un espejo disponible en GitHub. Los datos
se usan para estimar parámetros de modelos de Vasicek y generar 1000
simulaciones correlacionadas para 35 años con frecuencia mensual. También se
descarga una serie de tipo de cambio USD/CLP desde otro repositorio público.

La parte final implementa una red adversaria sencilla que busca optimizar las
ponderaciones de cada serie en el portafolio simulado.

**Limitaciones**
- El acceso directo a la Superintendencia de Pensiones está restringido,
  por lo que se utilizan archivos replicados en GitHub.
- El entorno usado para generar este ejemplo no cuenta con Python ni paquetes
  instalados, por lo que el código no se ha podido ejecutar ni probar.
