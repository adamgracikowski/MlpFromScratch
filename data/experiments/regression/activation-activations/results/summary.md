# Badanie `activation-activations`

Wpływ funkcji aktywacji w warstwach ukrytych na regresję zbioru `activation` w czterech rozmiarach.

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
| `sigmoid` | `100` | 10/10 | 0 | 0,1773 (0,1029–0,2272) | 51,83 (47,05–68,16) | 3,512 (3,299–3,877) | 0,1562 (0,08766–0,2189) | 119,7 (108,2–157,5) | 1000,0 | 5,15 s |
| `sigmoid` | `500` | 10/10 | 0 | 0,04358 (0,03218–0,07147) | 34,83 (29,28–43,16) | 2,715 (2,516–3,008) | 0,0431 (0,02942–0,06796) | 83,2 (69,99–103,2) | 300,0 | 2,49 s |
| `sigmoid` | `1000` | 10/10 | 0 | 0,04439 (0,02436–0,1299) | 25,89 (21,94–29,15) | 2,378 (2,174–2,498) | 0,04338 (0,02457–0,127) | 61,68 (52,3–69,52) | 200,0 | 2,31 s |
| `sigmoid` | `10000` | 10/10 | 0 | 0,03223 (0,01111–0,07878) | 15,92 (12,77–19,36) | 1,904 (1,609–2,038) | 0,03188 (0,01066–0,07825) | 37,98 (30,52–46,22) | 30,0 | 2,64 s |
| `tanh` | `100` | 10/10 | 0 | 0,0407 (0,01604–0,1495) | 1,017 (0,4555–2,39) | 0,6193 (0,4767–0,939) | 0,03716 (0,01579–0,1397) | 2,329 (0,8311–5,032) | 1000,0 | 4,76 s |
| `tanh` | `500` | 10/10 | 0 | 0,02087 (0,008652–0,04576) | 0,8839 (0,5111–1,725) | 0,5561 (0,4052–0,7609) | 0,02081 (0,007772–0,0455) | 2,064 (1,198–4,097) | 300,0 | 2,55 s |
| `tanh` | `1000` | 10/10 | 0 | 0,03238 (0,008776–0,0588) | 0,5309 (0,1032–0,7822) | 0,5075 (0,2323–0,599) | 0,03234 (0,008413–0,05823) | 1,174 (0,2404–1,787) | 200,0 | 2,37 s |
| `tanh` | `10000` | 10/10 | 0 | 0,02565 (0,01361–0,07864) | 0,2922 (0,2174–0,3537) | 0,3742 (0,2975–0,4363) | 0,02534 (0,01318–0,07837) | 0,5999 (0,3373–0,8237) | 30,0 | 2,72 s |
| ReLU | `100` | 10/10 | 0 | 2,859 (1,004–4,57) | 907,8 (856,3–1 002,3) | 14,23 (13,47–15,51) | 2,631 (1,299–3,756) | 2 097,5 (1 980,9–2 315,7) | 1000,0 | 2,5 s |
| ReLU | `500` | 10/10 | 0 | 0,3539 (0,1717–0,9266) | 669,4 (639,7–701,1) | 11,77 (11,48–11,91) | 0,3666 (0,1783–0,9576) | 1 600,7 (1 527,7–1 676,7) | 300,0 | 1,29 s |
| ReLU | `1000` | 10/10 | 0 | 1,143 (0,6786–1,697) | 650,8 (576,1–705) | 11,61 (10,59–12,47) | 1,152 (0,6864–1,699) | 1 552,4 (1 375,1–1 680,2) | 200,0 | 1,55 s |
| ReLU | `10000` | 10/10 | 0 | 0,2666 (0,1299–0,7962) | 591 (541,3–659,8) | 10,68 (10,28–11,43) | 0,2687 (0,1289–0,7999) | 1 413,5 (1 294,1–1 578,3) | 30,0 | 1,94 s |
| leaky ReLU | `100` | 10/10 | 0 | 0,9816 (0,7812–1,155) | 949,5 (909,6–1 085,0) | 14,51 (13,66–15,57) | 0,908 (0,6859–1,221) | 2 196,3 (2 104,1–2 509,8) | 1000,0 | 2,43 s |
| leaky ReLU | `500` | 10/10 | 0 | 0,7619 (0,2891–1,931) | 753,1 (714,3–790,8) | 12,67 (11,82–13,6) | 0,7574 (0,2641–1,81) | 1 800,6 (1 705,9–1 889,7) | 300,0 | 1,59 s |
| leaky ReLU | `1000` | 10/10 | 0 | 0,4286 (0,2993–0,6189) | 672,3 (625,7–786,2) | 11,72 (11,42–12,34) | 0,4078 (0,3109–0,6099) | 1 604,6 (1 493,5–1 876,6) | 200,0 | 1,58 s |
| leaky ReLU | `10000` | 10/10 | 0 | 0,2969 (0,09614–0,547) | 615,4 (575,5–685,6) | 11,21 (10,87–11,59) | 0,2974 (0,09601–0,5502) | 1 471,5 (1 375,1–1 639,9) | 30,0 | 1,95 s |

Mediana (w nawiasie kwartyle Q1–Q3) po ziarnach: błędy między ziarnami mają długi ogon i kilka złych ziaren zawyża średnią; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno. MSE i MAE w oryginalnej skali celu.

MSE w zakresie i poza zakresem: punkty testowe, których wejścia leżą w zakresie (od minimum do maksimum) wejść zbioru uczącego runu, i pozostałe; na wykresach `seeds_inside`, `heatmap_inside` (interpolacja) i `seeds_outside`, `heatmap_outside` (ekstrapolacja).

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `sigmoid__100` | `runs/sigmoid__100/seed-6` |
| `sigmoid__500` | `runs/sigmoid__500/seed-1` |
| `sigmoid__1000` | `runs/sigmoid__1000/seed-1` |
| `sigmoid__10000` | `runs/sigmoid__10000/seed-6` |
| `tanh__100` | `runs/tanh__100/seed-2` |
| `tanh__500` | `runs/tanh__500/seed-7` |
| `tanh__1000` | `runs/tanh__1000/seed-2` |
| `tanh__10000` | `runs/tanh__10000/seed-4` |
| `relu__100` | `runs/relu__100/seed-1` |
| `relu__500` | `runs/relu__500/seed-2` |
| `relu__1000` | `runs/relu__1000/seed-1` |
| `relu__10000` | `runs/relu__10000/seed-6` |
| `leaky-relu__100` | `runs/leaky-relu__100/seed-1` |
| `leaky-relu__500` | `runs/leaky-relu__500/seed-2` |
| `leaky-relu__1000` | `runs/leaky-relu__1000/seed-2` |
| `leaky-relu__10000` | `runs/leaky-relu__10000/seed-1` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
