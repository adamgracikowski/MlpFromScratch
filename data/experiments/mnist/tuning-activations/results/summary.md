# Badanie `tuning-activations`

Wpływ funkcji aktywacji warstwy ukrytej na klasyfikację MNIST: sieć 784-512-10, podzbiór 10000 obrazów uczących, wybór na zbiorze walidacyjnym.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `activation` | `model.hidden_activation`, `model.init` | `sigmoid`, `tanh`, `relu`, `leaky_relu` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5 |
| Liczba runów | 20 (4 x 5) |
| Porównanie wariantów | zbiór walidacyjny (metryka walidacyjna) |

## Konfiguracja bazowa

| Parametr | Wartość |
|---|---|
| `task` | `classification` |
| `data.train` | `data/datasets/mnist/data.mnist.train.60000.csv` |
| `data.test` | `data/datasets/mnist/data.mnist.test.10000.csv` |
| `data.target` | `label` |
| `data.input_normalization` | `min_max` |
| `data.target_normalization` | `none` |
| `data.train_subset` | 10000 |
| `data.validation_fraction` | 0,2 |
| `model.hidden_layers` | `[512]` |
| `model.bias` | `true` |
| `model.hidden_activation` | zmieniane przez oś `activation` |
| `model.output_activation` | `softmax` |
| `model.init` | zmieniane przez oś `activation` |
| `optimizer` | `adam` (`learning_rate` = 0,001) |
| `training.loss` | `cross_entropy` |
| `training.epochs` | 30 |
| `training.batch_size` | 32 |
| `training.early_stopping.patience` | 5 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 250 |
| `logging.max_weights_per_layer` | 64 |
| `output.grid_resolution` | 100 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Funkcja aktywacji | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (walidacyjny) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| `sigmoid` | 5/5 | 0 | 99,69% ± 0,35 | 94,08% ± 0,65 | 94,73% ± 0,29 | 23,8 | 154 s |
| `tanh` | 5/5 | 0 | 99,77% ± 0,24 | 94,01% ± 0,56 | 94,85% ± 0,25 | 17,0 | 127 s |
| ReLU | 5/5 | 0 | 99,77% ± 0,13 | 95,16% ± 0,39 | 95,68% ± 0,18 | 11,4 | 65,5 s |
| leaky ReLU | 5/5 | 0 | 99,76% ± 0,14 | 95,12% ± 0,48 | 95,60% ± 0,13 | 11,4 | 83,4 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `sigmoid` | `runs/sigmoid/seed-4` |
| `tanh` | `runs/tanh/seed-4` |
| `relu` | `runs/relu/seed-2` |
| `leaky_relu` | `runs/leaky_relu/seed-2` |

Dla każdego wariantu run, którego metryka walidacyjna jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
