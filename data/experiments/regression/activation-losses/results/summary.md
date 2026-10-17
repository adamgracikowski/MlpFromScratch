# Badanie `activation-losses`

Wpływ funkcji straty na regresję zbioru `activation` w czterech rozmiarach.

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
| MSE | `100` | 10/10 | 0 | 0,0407 (0,01604–0,1495) | 1,017 (0,4555–2,39) | 0,6193 (0,4767–0,939) | 0,03716 (0,01579–0,1397) | 2,329 (0,8311–5,032) | 1000,0 | 4,48 s |
| MSE | `500` | 10/10 | 0 | 0,02087 (0,008652–0,04576) | 0,8839 (0,5111–1,725) | 0,5561 (0,4052–0,7609) | 0,02081 (0,007772–0,0455) | 2,064 (1,198–4,097) | 300,0 | 2,3 s |
| MSE | `1000` | 10/10 | 0 | 0,03238 (0,008776–0,0588) | 0,5309 (0,1032–0,7822) | 0,5075 (0,2323–0,599) | 0,03234 (0,008413–0,05823) | 1,174 (0,2404–1,787) | 200,0 | 2,25 s |
| MSE | `10000` | 10/10 | 0 | 0,02565 (0,01361–0,07864) | 0,2922 (0,2174–0,3537) | 0,3742 (0,2975–0,4363) | 0,02534 (0,01318–0,07837) | 0,5999 (0,3373–0,8237) | 30,0 | 2,69 s |
| MAE | `100` | 10/10 | 0 | 0,8795 (0,5471–1,811) | 3,374 (1,9–4,328) | 1,27 (1,159–1,745) | 0,8409 (0,5073–1,639) | 7,023 (2,916–8,848) | 1000,0 | 3,94 s |
| MAE | `500` | 10/10 | 0 | 0,4281 (0,3474–0,5353) | 3,123 (1,048–3,761) | 1,189 (0,7702–1,441) | 0,4224 (0,3402–0,5447) | 5,934 (2,338–8,249) | 300,0 | 2,35 s |
| MAE | `1000` | 10/10 | 0 | 0,4634 (0,1816–0,6025) | 1,524 (0,6396–2,129) | 0,763 (0,6345–0,9946) | 0,473 (0,1781–0,6185) | 3,293 (0,669–4,577) | 200,0 | 2,39 s |
| MAE | `10000` | 10/10 | 0 | 0,3836 (0,2606–0,7753) | 1,879 (1,218–2,824) | 1,022 (0,898–1,303) | 0,3846 (0,2607–0,7774) | 3,398 (1,423–6,199) | 30,0 | 2,63 s |

Mediana (w nawiasie kwartyle Q1–Q3) po ziarnach: błędy między ziarnami mają długi ogon i kilka złych ziaren zawyża średnią; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno. MSE i MAE w oryginalnej skali celu.

MSE w zakresie i poza zakresem: punkty testowe, których wejścia leżą w zakresie (od minimum do maksimum) wejść zbioru uczącego runu, i pozostałe; na wykresach `seeds_inside`, `heatmap_inside` (interpolacja) i `seeds_outside`, `heatmap_outside` (ekstrapolacja).

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `mse__100` | `runs/mse__100/seed-2` |
| `mse__500` | `runs/mse__500/seed-7` |
| `mse__1000` | `runs/mse__1000/seed-2` |
| `mse__10000` | `runs/mse__10000/seed-4` |
| `mae__100` | `runs/mae__100/seed-6` |
| `mae__500` | `runs/mae__500/seed-4` |
| `mae__1000` | `runs/mae__1000/seed-3` |
| `mae__10000` | `runs/mae__10000/seed-6` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
