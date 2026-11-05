# EDA — UCI Household Electric Power

| Métrica | Valor |
| :--- | :--- |
| Filas (1 min) | 2,075,259 |
| Rango temporal | 2006-12-16 17:24:00 → 2010-11-26 21:02:00 |
| Faltantes (?→NaN) | 25,979 (1.25%) |
| Huecos >1 min | 0 (máx 0.0 h) |
| Media/Mediana kW | 1.092 / 0.602 |
| P95/Max kW | 3.264 / 11.122 |

Hora pico de consumo: **20:00 h** · Día pico: **Sáb**

## decisiones para Fase 2
Target: Global_active_power (kW) · resampleo a 1 h (media) ·
huecos: interpolación temporal + ffill · splits train/val/test 70/15/15
