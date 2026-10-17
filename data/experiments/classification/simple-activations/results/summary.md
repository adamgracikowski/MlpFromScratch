# Badanie `simple-activations`

Wpływ funkcji aktywacji w warstwach ukrytych na klasyfikację zbioru `simple` w czterech rozmiarach.

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
| `data.test` | `data/datasets/classification/data.simple.test.10000.csv` |
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
| `sigmoid` | `100` | 10/10 | 0 | 100,00% ± 0,00 | 99,24% ± 0,10 | 1000,0 | 18,8 s |
| `sigmoid` | `500` | 10/10 | 0 | 99,66% ± 0,13 | 99,36% ± 0,05 | 300,0 | 6,54 s |
| `sigmoid` | `1000` | 10/10 | 0 | 99,57% ± 0,29 | 99,50% ± 0,22 | 200,0 | 5,22 s |
| `sigmoid` | `10000` | 10/10 | 0 | 99,51% ± 0,12 | 99,46% ± 0,16 | 30,0 | 3,21 s |
| `tanh` | `100` | 10/10 | 0 | 100,00% ± 0,00 | 98,97% ± 0,03 | 1000,0 | 21,3 s |
| `tanh` | `500` | 10/10 | 0 | 99,70% ± 0,14 | 99,08% ± 0,18 | 300,0 | 8,71 s |
| `tanh` | `1000` | 10/10 | 0 | 99,46% ± 0,32 | 99,38% ± 0,37 | 200,0 | 6,18 s |
| `tanh` | `10000` | 10/10 | 0 | 99,44% ± 0,16 | 99,40% ± 0,19 | 30,0 | 4,03 s |
| ReLU | `100` | 10/10 | 0 | 100,00% ± 0,00 | 99,07% ± 0,12 | 1000,0 | 9,51 s |
| ReLU | `500` | 10/10 | 0 | 99,74% ± 0,10 | 99,16% ± 0,16 | 300,0 | 3,61 s |
| ReLU | `1000` | 10/10 | 0 | 99,44% ± 0,27 | 99,42% ± 0,22 | 200,0 | 2,75 s |
| ReLU | `10000` | 10/10 | 0 | 99,48% ± 0,15 | 99,40% ± 0,17 | 30,0 | 2,41 s |
| leaky ReLU | `100` | 10/10 | 0 | 100,00% ± 0,00 | 99,08% ± 0,11 | 1000,0 | 9,34 s |
| leaky ReLU | `500` | 10/10 | 0 | 99,78% ± 0,06 | 99,12% ± 0,12 | 300,0 | 3,84 s |
| leaky ReLU | `1000` | 10/10 | 0 | 99,42% ± 0,33 | 99,40% ± 0,27 | 200,0 | 2,65 s |
| leaky ReLU | `10000` | 10/10 | 0 | 99,47% ± 0,17 | 99,39% ± 0,18 | 30,0 | 2,18 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `sigmoid__100` | `runs/sigmoid__100/seed-1` |
| `sigmoid__500` | `runs/sigmoid__500/seed-3` |
| `sigmoid__1000` | `runs/sigmoid__1000/seed-1` |
| `sigmoid__10000` | `runs/sigmoid__10000/seed-2` |
| `tanh__100` | `runs/tanh__100/seed-5` |
| `tanh__500` | `runs/tanh__500/seed-3` |
| `tanh__1000` | `runs/tanh__1000/seed-1` |
| `tanh__10000` | `runs/tanh__10000/seed-2` |
| `relu__100` | `runs/relu__100/seed-9` |
| `relu__500` | `runs/relu__500/seed-3` |
| `relu__1000` | `runs/relu__1000/seed-2` |
| `relu__10000` | `runs/relu__10000/seed-1` |
| `leaky-relu__100` | `runs/leaky-relu__100/seed-9` |
| `leaky-relu__500` | `runs/leaky-relu__500/seed-3` |
| `leaky-relu__1000` | `runs/leaky-relu__1000/seed-2` |
| `leaky-relu__10000` | `runs/leaky-relu__10000/seed-1` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
