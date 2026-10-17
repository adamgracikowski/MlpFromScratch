# Badanie `simple-losses`

Wpływ funkcji straty na klasyfikację zbioru `simple` w czterech rozmiarach.

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
| `data.test` | `data/datasets/classification/data.simple.test.10000.csv` |
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
| entropia krzyżowa | `100` | 10/10 | 0 | 100,00% ± 0,00 | 98,97% ± 0,03 | 1000,0 | 12,6 s |
| entropia krzyżowa | `500` | 10/10 | 0 | 99,70% ± 0,14 | 99,08% ± 0,18 | 300,0 | 4,77 s |
| entropia krzyżowa | `1000` | 10/10 | 0 | 99,46% ± 0,32 | 99,38% ± 0,37 | 200,0 | 3,41 s |
| entropia krzyżowa | `10000` | 10/10 | 0 | 99,44% ± 0,16 | 99,40% ± 0,19 | 30,0 | 2,88 s |
| MSE | `100` | 10/10 | 0 | 100,00% ± 0,00 | 98,94% ± 0,02 | 1000,0 | 11,2 s |
| MSE | `500` | 10/10 | 0 | 99,70% ± 0,19 | 99,12% ± 0,31 | 300,0 | 3,98 s |
| MSE | `1000` | 10/10 | 0 | 99,60% ± 0,29 | 99,43% ± 0,17 | 200,0 | 2,77 s |
| MSE | `10000` | 10/10 | 0 | 99,42% ± 0,15 | 99,40% ± 0,18 | 30,0 | 2,54 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `cross_entropy__100` | `runs/cross_entropy__100/seed-5` |
| `cross_entropy__500` | `runs/cross_entropy__500/seed-3` |
| `cross_entropy__1000` | `runs/cross_entropy__1000/seed-1` |
| `cross_entropy__10000` | `runs/cross_entropy__10000/seed-2` |
| `mse__100` | `runs/mse__100/seed-2` |
| `mse__500` | `runs/mse__500/seed-6` |
| `mse__1000` | `runs/mse__1000/seed-2` |
| `mse__10000` | `runs/mse__10000/seed-2` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
