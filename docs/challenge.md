# Challenge CENABAST - Documentacion

## Resumen ejecutivo

Este repositorio implementa un pipeline completo de forecasting de consumo
farmaceutico para CENABAST, cubriendo las 6 partes del challenge:

- **Modelo (Parte I)**: modelo de dos etapas (hurdle model) que supera el
  baseline por producto en ~39% de MAE, muy por encima del 25% exigido.
- **API (Parte II)**: endpoint `/predict` con validacion de productos
  desconocidos y fechas invalidas (ambos retornando 400), sirviendo
  predicciones en O(1) por request via historial precomputado en memoria.
- **Tests (Parte III)**: suite completa con cobertura para modelo y API,
  validada tanto localmente como en un pipeline de CI limpio (Linux,
  Python 3.11).
- **Deploy (Parte IV)**: API desplegada en Cloud Run, verificada en vivo
  via `/health`, con la URL configurada en el `Makefile`.
- **CI/CD (Parte V)**: pipeline en GitHub Actions que ejecuta tests en
  cada push y despliega automaticamente a Cloud Run al mergear a `main`.
- **Analisis logistico (Parte VI)**: propuesta de calculo de punto de
  reorden y cantidad de pedido a partir de las predicciones de consumo.

## Decisiones de diseno

- **Target**: se filtra `tipo_movimiento == "S"` como definicion de consumo.
  Mezclar entradas y salidas no tiene interpretacion de negocio valida.
- **Calendario denso**: cada producto se reindexa dia a dia en su rango
  observado; la ausencia de movimiento es consumo = 0, no un dato faltante.
- **Modelo de dos etapas (hurdle model)**: la demanda es intermitente
  (~69% de los dias sin consumo). Una regresion Poisson unica no puede
  asignar masa exacta a cero, generando error sistematico. Se separa en:
  un clasificador que predice si habra consumo ese dia, y un regresor
  Poisson (entrenado solo sobre dias con consumo > 0) que predice la
  magnitud. Este diseno mejoro el MAE de ~4.4 a ~3.3 frente al mismo
  feature set con un solo regresor.
- **Codificacion de `gtin` para el modelo**: `HistGradientBoosting`
  requiere que las features categoricas usen codigos ordinales pequenos
  (< 255), pero los GTIN son codigos de barra de 13 digitos. Se remapean
  a enteros 0..n-1 justo antes de entrenar/predecir, manteniendo el GTIN
  real en el resto del sistema (historial, API, tests).
- **Fuentes adicionales**: `preprocess()` lee `stock.csv` por ruta fija
  ademas del argumento `data`, para incorporar nivel de inventario
  disponible como feature (senal que ningun lag de ventas puede derivar).
- **Feature mas relevante**: `dias_desde_ultima_venta` (enfoque tipo
  Croston para demanda intermitente), confirmado via permutation importance
  como la variable de mayor peso predictivo del modelo.

## Despliegue en la nube

El despliegue se realiza construyendo la imagen Docker directamente en el
runner de CI (`docker build` + `docker push`) hacia el repositorio de
Artifact Registry ya provisionado para esta postulacion
(`challenge-repo`, region `southamerica-west1`), y desplegando esa imagen
ya construida a Cloud Run con `gcloud run deploy --image`. Se opto por
este enfoque, en vez de `gcloud run deploy --source` (que delega el build
a Cloud Build), porque la Service Account entregada tiene permisos
acotados a Cloud Run y al repositorio de Artifact Registry especifico de
la postulacion, sin permiso de administracion de Cloud Build ni de
creacion de nuevos repositorios — decision confirmada mediante diagnostico
directo de los recursos y permisos disponibles en el proyecto.

## Parte VI - De consumo a proximo pedido de reabastecimiento

El modelo predice consumo diario (`cantidad`). El problema real de CENABAST
es "cuando pedir y cuanto pedir". Propuesta:

1. **Punto de reorden (ROP)**:
   `ROP = d_promedio_lead_time * L + z * sigma_d * sqrt(L)`
   donde `L` es el lead time de reposicion en dias (a validar con logistica
   real), `d_promedio` y `sigma_d` se derivan de las predicciones diarias
   del modelo sobre el horizonte de `L` dias, y `z` es el factor de nivel
   de servicio deseado (ej. z=1.65 para 95% de cobertura).

2. **Proyeccion de stock**: partiendo del `stock` actual (`stock.csv`),
   se resta dia a dia el consumo predicho hasta que
   `stock_proyectado <= ROP`. Esa fecha es el proximo pedido sugerido.

3. **Cantidad a pedir**: politica order-up-to-level:
   `Q = S_objetivo - stock_en_ROP`, con `S_objetivo = ROP + stock_seguridad`.

Esto convierte la prediccion puntual de consumo en una recomendacion
operativa (fecha + cantidad de pedido), que es el problema real de negocio
detras del challenge.
