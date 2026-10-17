# Badanie `cube-activations`

Wpływ funkcji aktywacji w warstwach ukrytych na regresję zbioru `cube` w czterech rozmiarach.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `activation` | `model.hidden_activation`, `model.init` | `sigmoid`, `tanh`, `relu`, `leaky-relu` |
| `size` | `data.test`, `data.train`, `training.epochs` | `100`, `500`, `1000`, `10000` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| Liczba runów | 160 (16 x 10) |

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
| `model.hidden_activation` | zmieniane przez oś `activation` |
| `model.output_activation` | `identity` |
| `model.init` | zmieniane przez oś `activation` |
| `optimizer` | `adam` (`learning_rate` = 0,003) |
| `training.loss` | `mse` |
| `training.epochs` | zmieniane przez oś `size` |
| `training.batch_size` | 32 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 100 |
| `logging.max_weights_per_layer` | 32 |
| `output.grid_resolution` | 500 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Funkcja aktywacji | Liczność zbioru uczącego | Ukończone | Rozbieżne | MSE (uczący) | MSE (testowy) | MAE (testowy) | MSE w zakresie (testowy) | MSE poza zakresem (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|---|---|---|
| `sigmoid` | `100` | 10/10 | 0 | 7,564 (6,137–8,788) | 51 601,5 (49 639,1–55 150,6) | 108,8 (106,3–112,2) | 5,936 (5,233–7,224) | 120 797,4 (116 207,8–129 110,2) | 1000,0 | 5,24 s |
| `sigmoid` | `500` | 10/10 | 0 | 1,259 (0,6264–1,656) | 40 962,4 (40 424,4–45 113,6) | 92,27 (91,64–97,28) | 1,122 (0,5123–1,342) | 97 609,4 (96 327,3–107 498,9) | 300,0 | 2,84 s |
| `sigmoid` | `1000` | 10/10 | 0 | 0,431 (0,2705–0,6513) | 41 899,5 (40 323,5–45 631,9) | 92,6 (91,45–96,67) | 0,4403 (0,2955–0,7003) | 99 843,1 (96 087,0–108 737,3) | 200,0 | 2,46 s |
| `sigmoid` | `10000` | 10/10 | 0 | 0,2003 (0,07627–0,219) | 37 246,6 (35 072,7–39 857,5) | 84,96 (82,53–87,99) | 0,1958 (0,07243–0,2152) | 89 109,6 (83 908,7–95 355,4) | 30,0 | 2,77 s |
| `tanh` | `100` | 10/10 | 0 | 0,8363 (0,5917–3,088) | 40 190,2 (38 104,3–44 393,3) | 89,81 (87,04–94,71) | 0,6799 (0,5906–2,674) | 94 089,7 (89 186,5–103 917,5) | 1000,0 | 5,6 s |
| `tanh` | `500` | 10/10 | 0 | 0,6012 (0,5493–0,7653) | 35 009,9 (33 824,2–38 105,3) | 80,66 (80,01–84,72) | 0,5758 (0,4861–0,6455) | 83 425,7 (80 600,1–90 801,8) | 300,0 | 2,76 s |
| `tanh` | `1000` | 10/10 | 0 | 0,3798 (0,2136–1,526) | 39 669,9 (39 194,7–41 720,4) | 87,12 (86,52–89,57) | 0,4026 (0,2287–1,535) | 94 527,1 (93 397,6–99 416,6) | 200,0 | 2,73 s |
| `tanh` | `10000` | 10/10 | 0 | 0,2719 (0,1424–0,306) | 38 021,9 (36 488,5–40 625,5) | 83,88 (81,16–86,11) | 0,2663 (0,1401–0,2977) | 90 964,6 (87 295,8–97 193,5) | 30,0 | 3,07 s |
| ReLU | `100` | 10/10 | 0 | 1,822 (1,041–16,52) | 35 294,1 (33 030,3–36 904,2) | 86,94 (84,93–92,37) | 1,655 (1,076–15,75) | 82 626,4 (77 326,8–86 388,2) | 1000,0 | 1,85 s |
| ReLU | `500` | 10/10 | 0 | 1,186 (0,5846–2,162) | 30 680,6 (30 176,7–36 783,0) | 80,4 (78,82–87,15) | 1,144 (0,5–1,709) | 73 107,6 (71 908,4–87 650,0) | 300,0 | 1,33 s |
| ReLU | `1000` | 10/10 | 0 | 1,446 (0,7153–3,811) | 31 220,1 (27 909,4–35 120,1) | 83,31 (76,95–85,71) | 1,462 (0,7241–4,037) | 74 390,0 (66 501,4–83 687,4) | 200,0 | 1,39 s |
| ReLU | `10000` | 10/10 | 0 | 0,4703 (0,2657–0,9584) | 31 624,0 (26 811,8–35 201,5) | 80,83 (74,44–86,01) | 0,4629 (0,2583–0,9482) | 75 656,7 (64 144,9–84 214,3) | 30,0 | 1,87 s |
| leaky ReLU | `100` | 10/10 | 0 | 1,54 (0,8975–1,89) | 32 344,5 (29 621,3–33 874,5) | 82,95 (79,41–85,16) | 1,704 (0,8441–1,876) | 75 720,9 (69 345,1–79 301,6) | 1000,0 | 1,98 s |
| leaky ReLU | `500` | 10/10 | 0 | 0,4498 (0,2661–0,7429) | 26 121,3 (23 702,9–30 161,5) | 73,78 (69,7–78,58) | 0,3851 (0,2333–0,6619) | 62 244,7 (56 482,0–71 872,6) | 300,0 | 1,4 s |
| leaky ReLU | `1000` | 10/10 | 0 | 0,612 (0,3597–1,007) | 26 484,2 (22 739,3–31 947,1) | 74,53 (69,23–81,37) | 0,6472 (0,3683–1,054) | 63 109,0 (54 184,0–76 127,4) | 200,0 | 1,49 s |
| leaky ReLU | `10000` | 10/10 | 0 | 0,7718 (0,289–1,305) | 26 111,0 (21 769,3–30 771,0) | 73,62 (67,57–77,95) | 0,7532 (0,2851–1,28) | 62 467,4 (52 078,9–73 617,1) | 30,0 | 1,81 s |

Mediana (w nawiasie kwartyle Q1–Q3) po ziarnach: błędy między ziarnami mają długi ogon i kilka złych ziaren zawyża średnią; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno. MSE i MAE w oryginalnej skali celu.

MSE w zakresie i poza zakresem: punkty testowe, których wejścia leżą w zakresie (od minimum do maksimum) wejść zbioru uczącego runu, i pozostałe; na wykresach `seeds_inside`, `heatmap_inside` (interpolacja) i `seeds_outside`, `heatmap_outside` (ekstrapolacja).

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `sigmoid__100` | `runs/sigmoid__100/seed-6` |
| `sigmoid__500` | `runs/sigmoid__500/seed-1` |
| `sigmoid__1000` | `runs/sigmoid__1000/seed-1` |
| `sigmoid__10000` | `runs/sigmoid__10000/seed-5` |
| `tanh__100` | `runs/tanh__100/seed-2` |
| `tanh__500` | `runs/tanh__500/seed-2` |
| `tanh__1000` | `runs/tanh__1000/seed-2` |
| `tanh__10000` | `runs/tanh__10000/seed-2` |
| `relu__100` | `runs/relu__100/seed-8` |
| `relu__500` | `runs/relu__500/seed-5` |
| `relu__1000` | `runs/relu__1000/seed-1` |
| `relu__10000` | `runs/relu__10000/seed-1` |
| `leaky-relu__100` | `runs/leaky-relu__100/seed-4` |
| `leaky-relu__500` | `runs/leaky-relu__500/seed-3` |
| `leaky-relu__1000` | `runs/leaky-relu__1000/seed-2` |
| `leaky-relu__10000` | `runs/leaky-relu__10000/seed-4` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
