# Badanie `tuning-batch-sizes`

Wpływ rozmiaru batcha na klasyfikację MNIST: sieć 784-512-10 z ReLU, Adam (0,001), podzbiór 10000 obrazów uczących, wybór na zbiorze walidacyjnym.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `batch` | `training.batch_size` | `16`, `32`, `64`, `128` |

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
| `model.hidden_activation` | `relu` |
| `model.output_activation` | `softmax` |
| `model.init` | `he` |
| `optimizer` | `adam` (`learning_rate` = 0,001) |
| `training.loss` | `cross_entropy` |
| `training.epochs` | 30 |
| `training.batch_size` | zmieniane przez oś `batch` |
| `training.early_stopping.patience` | 5 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 250 |
| `logging.max_weights_per_layer` | 64 |
| `output.grid_resolution` | 100 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Rozmiar batcha | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (walidacyjny) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| `16` | 5/5 | 0 | 99,10% ± 0,93 | 94,81% ± 0,79 | 95,58% ± 0,45 | 10,2 | 98,3 s |
| `32` | 5/5 | 0 | 99,77% ± 0,13 | 95,16% ± 0,39 | 95,68% ± 0,18 | 11,4 | 88,3 s |
| `64` | 5/5 | 0 | 99,78% ± 0,21 | 94,92% ± 0,58 | 95,67% ± 0,22 | 14,0 | 87,7 s |
| `128` | 5/5 | 0 | 99,91% ± 0,07 | 94,83% ± 0,46 | 95,52% ± 0,09 | 16,8 | 76,2 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `16` | `runs/16/seed-2` |
| `32` | `runs/32/seed-2` |
| `64` | `runs/64/seed-2` |
| `128` | `runs/128/seed-3` |

Dla każdego wariantu run, którego metryka walidacyjna jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
