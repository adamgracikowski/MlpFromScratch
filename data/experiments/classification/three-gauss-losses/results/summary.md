# Badanie `three-gauss-losses`

Wpływ funkcji straty na klasyfikację zbioru `three_gauss` w czterech rozmiarach.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `loss` | `training.loss` | `cross_entropy`, `mse` |
| `size` | `data.train`, `training.epochs` | `100`, `500`, `1000`, `10000` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| Liczba runów | 80 (8 x 10) |

## Konfiguracja bazowa

| Parametr | Wartość |
|---|---|
| `task` | `classification` |
| `data.train` | zmieniane przez oś `size` |
| `data.test` | `data/datasets/classification/data.three_gauss.test.10000.csv` |
| `data.target` | `cls` |
| `data.input_normalization` | `standard` |
| `data.target_normalization` | `none` |
| `data.validation_fraction` | 0 |
| `model.hidden_layers` | `[16, 16]` |
| `model.bias` | `true` |
| `model.hidden_activation` | `tanh` |
| `model.output_activation` | `softmax` |
| `model.init` | `xavier` |
| `optimizer` | `adam` (`learning_rate` = 0,003) |
| `training.loss` | zmieniane przez oś `loss` |
| `training.epochs` | zmieniane przez oś `size` |
| `training.batch_size` | 32 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 100 |
| `logging.max_weights_per_layer` | 32 |
| `output.grid_resolution` | 100 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Funkcja straty | Liczność zbioru uczącego | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| entropia krzyżowa | `100` | 10/10 | 0 | 96,57% ± 0,32 | 92,72% ± 0,10 | 1000,0 | 60,9 s |
| entropia krzyżowa | `500` | 10/10 | 0 | 94,85% ± 0,13 | 93,29% ± 0,07 | 300,0 | 23,8 s |
| entropia krzyżowa | `1000` | 10/10 | 0 | 93,71% ± 0,12 | 93,51% ± 0,13 | 200,0 | 17 s |
| entropia krzyżowa | `10000` | 10/10 | 0 | 93,79% ± 0,08 | 93,64% ± 0,09 | 30,0 | 10,6 s |
| MSE | `100` | 10/10 | 0 | 96,63% ± 0,25 | 92,63% ± 0,12 | 1000,0 | 54,6 s |
| MSE | `500` | 10/10 | 0 | 95,03% ± 0,24 | 93,35% ± 0,05 | 300,0 | 21,2 s |
| MSE | `1000` | 10/10 | 0 | 93,83% ± 0,11 | 93,48% ± 0,10 | 200,0 | 13,8 s |
| MSE | `10000` | 10/10 | 0 | 93,76% ± 0,07 | 93,65% ± 0,11 | 30,0 | 8,97 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `cross_entropy__100` | `runs/cross_entropy__100/seed-2` |
| `cross_entropy__500` | `runs/cross_entropy__500/seed-1` |
| `cross_entropy__1000` | `runs/cross_entropy__1000/seed-1` |
| `cross_entropy__10000` | `runs/cross_entropy__10000/seed-7` |
| `mse__100` | `runs/mse__100/seed-5` |
| `mse__500` | `runs/mse__500/seed-7` |
| `mse__1000` | `runs/mse__1000/seed-1` |
| `mse__10000` | `runs/mse__10000/seed-4` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
