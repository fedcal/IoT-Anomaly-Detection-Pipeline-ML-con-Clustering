---
layout: default
title: Scelte tecniche
nav_order: 3
has_children: true
permalink: /scelte_tecniche/
description: >-
  Decisioni architetturali e di modellazione del progetto IoT Anomaly
  Detection Pipeline, con trade-off espliciti e razionali documentati.
---

# Scelte tecniche

Questa sezione documenta **come** è costruito il progetto e **perché** ogni
componente è stata progettata in un certo modo. È pensata per chi vuole
estendere o adattare la pipeline a un dominio simile (manutenzione predittiva,
monitoring industriale, sensoristica).

## Capitoli

| Capitolo | Titolo | Cosa contiene |
|:--|:--|:--|
| 1 | [Architettura](architettura/) | Moduli `src/iot_anomaly/`, flusso dati, CLI `iot-detect`, dipendenze fra componenti. |
| 2 | [Scelte di modellazione](scelte_modello/) | Famiglia di clustering, scelta di K, finestra rolling, soglia, gestione del rischio. |

{: .tip }
> Per la teoria sottostante (algoritmi, metriche, anti-leakage) consulta
> la sezione **[Teoria](../teoria/)**.
