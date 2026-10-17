# Badanie `cube-losses`

Wpływ funkcji straty na regresję zbioru `cube` w czterech rozmiarach.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `loss` | `training.loss` | `mse`, `mae` |
| `size` | `data.test`, `data.train`, `training.epochs` | `100`, `500`, `1000`, `10000` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| Liczba runów | 80 (8 x 10) |

## Konfiguracja bazowa

| Parametr | Wartość |
|---|---|
| `task` | `regression` |
| `data.train` | zmieniane przez oś `size` |
| `data.test` | zmieniane przez oś `size` |
| `data.target` | `y` |
| `data.input_normalization` | `standard` |
| `data.target_normalization` | `standard` |
| `data.validation_fraction` | 0 |
| `model.hidden_layers` | `[32, 32]` |
| `model.bias` | `true` |
| `model.hidden_activation` | `tanh` |
| `model.output_activation` | `identity` |
| `model.init` | `xavier` |
| `optimizer` | `adam` (`learning_rate` = 0,003) |
| `training.loss` | zmieniane przez oś `loss` |
| `training.epochs` | zmieniane przez oś `size` |
| `training.batch_size` | 32 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 100 |
| `logging.max_weights_per_layer` | 32 |
| `output.grid_resolution` | 500 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Funkcja straty | Liczność zbioru uczącego | Ukończone | Rozbieżne | MSE (uczący) | MSE (testowy) | MAE (testowy) | MSE w zakresie (testowy) | MSE poza zakresem (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|---|---|---|
| MSE | `100` | 10/10 | 0 | 0,8363 (0,5917–3,088) | 40 190,2 (38 104,3–44 393,3) | 89,81 (87,04–94,71) | 0,6799 (0,5906–2,674) | 94 089,7 (89 186,5–103 917,5) | 1000,0 | 4,69 s |
| MSE | `500` | 10/10 | 0 | 0,6012 (0,5493–0,7653) | 35 009,9 (33 824,2–38 105,3) | 80,66 (80,01–84,72) | 0,5758 (0,4861–0,6455) | 83 425,7 (80 600,1–90 801,8) | 300,0 | 2,42 s |
| MSE | `1000` | 10/10 | 0 | 0,3798 (0,2136–1,526) | 39 669,9 (39 194,7–41 720,4) | 87,12 (86,52–89,57) | 0,4026 (0,2287–1,535) | 94 527,1 (93 397,6–99 416,6) | 200,0 | 2,5 s |
| MSE | `10000` | 10/10 | 0 | 0,2719 (0,1424–0,306) | 38 021,9 (36 488,5–40 625,5) | 83,88 (81,16–86,11) | 0,2663 (0,1401–0,2977) | 90 964,6 (87 295,8–97 193,5) | 30,0 | 2,75 s |
| MAE | `100` | 10/10 | 0 | 6,606 (4,291–7,818) | 44 293,5 (43 033,1–45 467,6) | 97,22 (96,67–98,01) | 5,581 (3,869–6,501) | 103 687,8 (100 738,1–106 438,0) | 1000,0 | 4,75 s |
| MAE | `500` | 10/10 | 0 | 4,255 (3,162–5,931) | 32 595,0 (31 926,7–33 113,3) | 81,78 (81,11–82,3) | 3,402 (2,547–5,254) | 77 667,8 (76 072,3–78 903,6) | 300,0 | 2,34 s |
| MAE | `1000` | 10/10 | 0 | 3,152 (2,584–4,009) | 39 745,7 (38 043,9–41 740,4) | 90,32 (89,38–91,71) | 3,253 (2,798–4,284) | 94 707,4 (90 652,0–99 461,8) | 200,0 | 2,18 s |
| MAE | `10000` | 10/10 | 0 | 1,972 (1,574–2,241) | 39 332,7 (35 252,0–40 965,4) | 87,9 (84–89,6) | 1,931 (1,522–2,208) | 94 098,1 (84 334,0–98 004,5) | 30,0 | 2,7 s |

Mediana (w nawiasie kwartyle Q1–Q3) po ziarnach: błędy między ziarnami mają długi ogon i kilka złych ziaren zawyża średnią; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno. MSE i MAE w oryginalnej skali celu.

MSE w zakresie i poza zakresem: punkty testowe, których wejścia leżą w zakresie (od minimum do maksimum) wejść zbioru uczącego runu, i pozostałe; na wykresach `seeds_inside`, `heatmap_inside` (interpolacja) i `seeds_outside`, `heatmap_outside` (ekstrapolacja).

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `mse__100` | `runs/mse__100/seed-2` |
| `mse__500` | `runs/mse__500/seed-2` |
| `mse__1000` | `runs/mse__1000/seed-2` |
| `mse__10000` | `runs/mse__10000/seed-2` |
| `mae__100` | `runs/mae__100/seed-6` |
| `mae__500` | `runs/mae__500/seed-1` |
| `mae__1000` | `runs/mae__1000/seed-4` |
| `mae__10000` | `runs/mae__10000/seed-4` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
