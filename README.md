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
