# Challenge CENABAST - Documentacion

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
- **Fuentes adicionales**: `preprocess()` lee `stock.csv` por ruta fija
  ademas del argumento `data`, para incorporar nivel de inventario
  disponible como feature (senal que ningun lag de ventas puede derivar).
- **Feature mas relevante**: `dias_desde_ultima_venta` (enfoque tipo
  Croston para demanda intermitente), confirmado via permutation importance
  como la variable de mayor peso predictivo del modelo.

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
