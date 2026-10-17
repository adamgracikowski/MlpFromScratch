# Badanie `three-gauss-activations`

Wpływ funkcji aktywacji w warstwach ukrytych na klasyfikację zbioru `three_gauss` w czterech rozmiarach.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `activation` | `model.hidden_activation`, `model.init` | `sigmoid`, `tanh`, `relu`, `leaky-relu` |
| `size` | `data.train`, `training.epochs` | `100`, `500`, `1000`, `10000` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| Liczba runów | 160 (16 x 10) |

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
| `model.hidden_activation` | zmieniane przez oś `activation` |
| `model.output_activation` | `softmax` |
| `model.init` | zmieniane przez oś `activation` |
| `optimizer` | `adam` (`learning_rate` = 0,003) |
| `training.loss` | `cross_entropy` |
| `training.epochs` | zmieniane przez oś `size` |
| `training.batch_size` | 32 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 100 |
| `logging.max_weights_per_layer` | 32 |
| `output.grid_resolution` | 100 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Funkcja aktywacji | Liczność zbioru uczącego | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| `sigmoid` | `100` | 10/10 | 0 | 95,27% ± 0,26 | 92,88% ± 0,07 | 1000,0 | 52,8 s |
| `sigmoid` | `500` | 10/10 | 0 | 94,57% ± 0,14 | 93,52% ± 0,07 | 300,0 | 20,4 s |
| `sigmoid` | `1000` | 10/10 | 0 | 93,48% ± 0,12 | 93,70% ± 0,05 | 200,0 | 14,1 s |
| `sigmoid` | `10000` | 10/10 | 0 | 93,79% ± 0,06 | 93,69% ± 0,07 | 30,0 | 9,59 s |
| `tanh` | `100` | 10/10 | 0 | 96,57% ± 0,32 | 92,72% ± 0,10 | 1000,0 | 76,4 s |
| `tanh` | `500` | 10/10 | 0 | 94,85% ± 0,13 | 93,29% ± 0,07 | 300,0 | 23,6 s |
| `tanh` | `1000` | 10/10 | 0 | 93,71% ± 0,12 | 93,51% ± 0,13 | 200,0 | 21,1 s |
| `tanh` | `10000` | 10/10 | 0 | 93,79% ± 0,08 | 93,64% ± 0,09 | 30,0 | 10,8 s |
| ReLU | `100` | 10/10 | 0 | 97,00% ± 0,31 | 92,47% ± 0,31 | 1000,0 | 31,6 s |
| ReLU | `500` | 10/10 | 0 | 95,01% ± 0,23 | 93,29% ± 0,08 | 300,0 | 10,6 s |
| ReLU | `1000` | 10/10 | 0 | 93,99% ± 0,14 | 93,39% ± 0,15 | 200,0 | 8,46 s |
| ReLU | `10000` | 10/10 | 0 | 93,87% ± 0,05 | 93,69% ± 0,10 | 30,0 | 8,38 s |
| leaky ReLU | `100` | 10/10 | 0 | 97,17% ± 0,28 | 92,56% ± 0,28 | 1000,0 | 31,1 s |
| leaky ReLU | `500` | 10/10 | 0 | 95,13% ± 0,29 | 93,26% ± 0,10 | 300,0 | 10,8 s |
| leaky ReLU | `1000` | 10/10 | 0 | 93,96% ± 0,14 | 93,36% ± 0,15 | 200,0 | 8,1 s |
| leaky ReLU | `10000` | 10/10 | 0 | 93,87% ± 0,04 | 93,67% ± 0,07 | 30,0 | 7,54 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `sigmoid__100` | `runs/sigmoid__100/seed-2` |
| `sigmoid__500` | `runs/sigmoid__500/seed-2` |
| `sigmoid__1000` | `runs/sigmoid__1000/seed-3` |
| `sigmoid__10000` | `runs/sigmoid__10000/seed-6` |
| `tanh__100` | `runs/tanh__100/seed-2` |
| `tanh__500` | `runs/tanh__500/seed-1` |
| `tanh__1000` | `runs/tanh__1000/seed-1` |
| `tanh__10000` | `runs/tanh__10000/seed-7` |
| `relu__100` | `runs/relu__100/seed-4` |
| `relu__500` | `runs/relu__500/seed-4` |
| `relu__1000` | `runs/relu__1000/seed-3` |
| `relu__10000` | `runs/relu__10000/seed-8` |
| `leaky-relu__100` | `runs/leaky-relu__100/seed-3` |
| `leaky-relu__500` | `runs/leaky-relu__500/seed-4` |
| `leaky-relu__1000` | `runs/leaky-relu__1000/seed-4` |
| `leaky-relu__10000` | `runs/leaky-relu__10000/seed-3` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
