# Badanie `tuning-normalization`

Wpływ normalizacji wejść (pikseli) na klasyfikację MNIST: podzbiór 10000 obrazów uczących, wybór na zbiorze walidacyjnym.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `normalization` | `data.input_normalization` | `min_max`, `standard` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5 |
| Liczba runów | 10 (2 x 5) |
| Porównanie wariantów | zbiór walidacyjny (metryka walidacyjna) |

## Konfiguracja bazowa

| Parametr | Wartość |
|---|---|
| `task` | `classification` |
| `data.train` | `data/datasets/mnist/data.mnist.train.60000.csv` |
| `data.test` | `data/datasets/mnist/data.mnist.test.10000.csv` |
| `data.target` | `label` |
| `data.input_normalization` | zmieniane przez oś `normalization` |
| `data.target_normalization` | `none` |
| `data.train_subset` | 10000 |
| `data.validation_fraction` | 0,2 |
| `model.hidden_layers` | `[128]` |
| `model.bias` | `true` |
| `model.hidden_activation` | `relu` |
| `model.output_activation` | `softmax` |
| `model.init` | `he` |
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

| Normalizacja wejść | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (walidacyjny) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| min-max | 5/5 | 0 | 99,47% ± 0,65 | 94,35% ± 0,61 | 95,18% ± 0,27 | 15,2 | 4,26 s |
| standaryzacja | 5/5 | 0 | 99,64% ± 0,52 | 93,90% ± 0,18 | 94,39% ± 0,26 | 10,4 | 2,9 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `min_max` | `runs/min_max/seed-2` |
| `standard` | `runs/standard/seed-5` |

Dla każdego wariantu run, którego metryka walidacyjna jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
