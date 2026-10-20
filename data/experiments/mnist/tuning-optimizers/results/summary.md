# Badanie `tuning-optimizers`

Wpływ optymalizatora i współczynnika uczenia na klasyfikację MNIST: sieć 784-512-10 z ReLU, podzbiór 10000 obrazów uczących, wybór na zbiorze walidacyjnym.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `optimizer` | `optimizer` | `sgd-0.1`, `momentum-0.01`, `adam-0.0003`, `adam-0.001`, `adam-0.003` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5 |
| Liczba runów | 25 (5 x 5) |
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
| `optimizer` | zmieniane przez oś `optimizer` |
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

| Optymalizator | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (walidacyjny) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| SGD (0,1) | 5/5 | 0 | 99,77% ± 0,11 | 94,54% ± 0,66 | 95,30% ± 0,30 | 20,4 | 22 s |
| momentum (0,01) | 5/5 | 0 | 99,78% ± 0,23 | 94,47% ± 0,79 | 95,34% ± 0,27 | 22,0 | 26 s |
| Adam (0,0003) | 5/5 | 0 | 99,80% ± 0,22 | 94,59% ± 0,73 | 95,41% ± 0,28 | 20,2 | 26,6 s |
| Adam (0,001) | 5/5 | 0 | 99,77% ± 0,13 | 95,16% ± 0,39 | 95,68% ± 0,18 | 11,4 | 15,3 s |
| Adam (0,003) | 5/5 | 0 | 98,79% ± 0,66 | 94,71% ± 0,56 | 95,21% ± 0,49 | 8,8 | 11,9 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `sgd-0.1` | `runs/sgd-0.1/seed-5` |
| `momentum-0.01` | `runs/momentum-0.01/seed-1` |
| `adam-0.0003` | `runs/adam-0.0003/seed-1` |
| `adam-0.001` | `runs/adam-0.001/seed-2` |
| `adam-0.003` | `runs/adam-0.003/seed-2` |

Dla każdego wariantu run, którego metryka walidacyjna jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
