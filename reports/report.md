# Fizyka Ogólna - projekt nr 65 - Sprawozdanie
<!-- `pandoc -o reports/report.pdf reports/report.md -V geometry:margin=0.5in -V lang=polish -f markdown+raw_tex -V graphics=true` -->

* Mikołaj Garbowski
* Maksym Bieńkowski

## Opis zadania
Temat zadania to *AI do klasyfikacji obiektów astronomicznych na podstawie danych fotometrycznych* (oryginalnie: do klasyfikacji typów supernowy, końcowo wybraliśmy jednak zestaw danych z większą ilością klas). 

## Dane wejściowe
Zdecydowaliśmy się skorzystać z zestawu danych z konkursu [PLAsTiCC 2018](https://www.kaggle.com/competitions/PLAsTiCC-2018) zorganizowanego przez członków kolaboracji [LSST](https://lsst-tvssc.github.io/) w przygotowaniu na ogromne wolumeny danych fotometrycznych z prowadzonych pomiarów. Zestaw jest ogromny - całość waży ponad 40GB, w związku z czym działaliśmy na jego części zawierającej dane o ponad 7800 obiektach.

Zestaw podzielony był na dwie części - szeregi czasowe zawierające dane dotyczące światła obiektów oraz metadane. Poniżej format danych:

### Metadane obiektów (plik nagłówkowy, jeden wiersz = jeden obiekt)
- `object_id`: unikalny identyfikator obiektu  
- `ra`: rektascensja, współrzędna nieba (stopnie)  
- `decl`: deklinacja, współrzędna nieba (stopnie)  
- `gal_l`: długość galaktyczna (stopnie)  
- `gal_b`: szerokość galaktyczna (stopnie)  
- `ddf`: flaga logiczna, czy obiekt pochodzi z obszaru Deep Drilling Field  
- `hostgal_specz`: spektroskopowe przesunięcie ku czerwieni obiektu (dokładny, głównie w zbiorze treningowym)  
- `hostgal_photoz`: fotometryczny przesunięcie ku czerwieni galaktyki macierzystej  
- `hostgal_photoz_err`: niepewność `hostgal_photoz`  
- `distmod`: moduł odległości obliczony z fotometrycznego przesunięcia ku czerwieni
- `MWEBV`: ekstynkcja pyłowa Drogi Mlecznej wzdłuż linii widzenia  
- `target`: etykieta klasy obiektu

### Dane szeregów czasowych (krzywe blasku, jeden wiersz = jedna obserwacja)
- `object_id`: identyfikator obiektu, klucz łączący z metadanymi  
- `mjd`: znacznik czasowy obserwacji (zmodyfikowana data w kalendarzu juliańskim)
- `passband`: pasmo obserwacyjne LSST (u/g/r/i/z/y)
- `flux`: zmierzony strumień jasności w danym filtrze  
- `flux_err`: niepewność pomiaru strumienia  
- `detected`: flaga logiczna wykrycia istotnego statystycznie ($\le 3\sigma$)

### Uwagi
- obserwacje w różnych filtrach nie były wykonywane jednocześnie  
- obserwacje nie były wykonywane w równomiernych odstępach 

### Rozkład klas

![Histogram rozkładu klas](reports/assets/class_distribution.png){height=400px}


## Architektura modelu

Pojedyncze wejście modelu składa się z metadanych obiektu i sześciu szeregów czasowych (po jednym na pasmo).
Metadane są przekształcane przez perceptron wielowarstwowy do wektora liczb rzeczywistych.
Każdy szereg czasowy jest przetwarzany przez oddzielną sięć LSTM, która na wyjściu zwraca wektor liczb rzeczywistych reprezentujący ten szereg.
Połączone wyjścia perceptronu i sześciu sieci LSTM stanowią wejście głowicy klasyfikacyjnej (również perceptron wielowarstwowy),
która zwraca logity - nieznormalizowane prawdopodobieństwa przynależności do poszczególnych klas.

Mając do klasyfikacji dane będące szeregami czasowymi, zdecydowaliśmy się użyć sieci LSTM (Hochreiter, 1997) ze względu na ich wysoką skuteczność i niską liczbę parametrów w porównaniu do bardziej złożonych rozwiązań, np. transformerów.

![diagram architektury modelu](reports/assets/architecture-diagram.drawio.svg)

## Przetwarzanie danych
Przed wytrenowaniem i ewaluacją modelu przetworzyliśmy dane, aby wycisnąć z nich jak najwięcej istotnych informacji. 

### Metadane - inżynieria cech
Usunęliśmy niepotrzebne kolumny:

* `ra`, `decl`, `gal_l` i `gal_b` - położenie obiektu na niebie nie powinno być skorelowane z jego typem, a dane pochodzą z symulacji. Istotna informacja o dystansie zawarta jest i tak w parametrze `distmod`.
* `hostgal_photoz`, `hostgal_photoz_err` - jako że zarówno trening, jak i ewaluacja przebiegał na etykietowanych danych treningowych, które zawierały dokładnie zmierzone spektroskopowe wartości przesunięcia ku czerwieni.
  * W nieetykietowanym zestawie testowym, na którym ewaluowano modele w oficjalnych zawodach, w większości pomiarów brakowało danych spektroskopowych, stąd włączenie danych fotometrycznych w zestaw. Na nasze potrzeby wystarczą więc dane spektroskopowe.

Dodaliśmy nowe kolumny:

* `n_obs` - łączna liczba obserwacji
* `n_detections` - łączna liczba obserwacji, gdy obiekt był wykryty
* `t_span` - łączny okres pomiaru, różnica czasu ostatniego i pierwszego pomiaru (w dniach)
* `max_snr_{0-5}` - heurystyka maksymalnej jasności w danym pasmie zawierająca informację o błędzie pomiaru (signal to noise ratio)
* `mean_flux_{0-5}` średnia wartość strumienia per pasmo we wszystkich pomiarach - uśredniona informacja o jasności

### Normalizacja metadanych

Na podstawie analizy rozkładów prawdopodobieństwa poszczególnych zmiennych stosowaliśmy w większości przypadków standardową normalizację $(x-\mu) / \sigma$. Jeśli rozkład zawierał dużo próbek o małych wartościach i długi, kilka rzędów większy ogon, stosowaliśmy transformację logarytmiczną $\mathrm{log1p}(x) = \log(1+x)$, aby uzyskać bardziej symetryczny rozkład do poddania normalizacji.

### Szeregi czasowe - inżynieria cech

Usunęliśmy kolumnę mjd - data w formacie bezwzględnym nie jest tu pomocna, w szczególności, jeśli dane testowe pochodzić będą z innego okresu, niż treningowe. Sieci LSTM powinny operować na względnych zmianach czasu w porównaniu do poprzedniego pomiaru.

Dodaliśmy nowe kolumny:

* `delta_t` - wspomniana względna zmiana czasu od ostatniego pomiaru, dla pierwszego pomiaru dla danego obiektu w danym paśmie równa 0
* `delta_t_cumsum` - czas, który upłynął od pierwszego pomiaru obiektu w danym paśmie
* `snr` - *signal to noise ratio* wyliczane jako `flux/flux_err` - korzystna informacja dla modelu opisująca pewność pomiaru

### Normalizacja szeregów czasowych

Normalizowane są wszystkie dane podawane do LSTMa - istotny jest w tym przypadku ich ogólny "kształt", a klasyfikator dostaje na wejście nieznormalizowane zagregowane dane z szeregów czasowych. Dla każdego ciągu obserwacji (per obiekt i pasmo) normalizowane są wartości `flux` i `flux_err` poprzez podzielenie przez medianę wartości bezwzględnej strumienia. Mediana zamiast wartości średniej sprawia, że zaszumione skoki nie wpływają na skalę uśredniania, a wartość bezwzględna nie zmienia znaku.

## Dobór hiperparametrów i trening

Aby znaleźć optymalne hiperparametry modelu, przeprowadziliśmy losowe przeszukiwanie przestrzeni ich wartości z wykorzystaniem serwisu [weights and biases](https://wandb.ai/). Wartości dokładności na zbiorze walidacyjnym wahały się w granicach 53-73%. Najlepszy model (o najwyższej dokładności na zbiorze walidacyjnym) został wykorzystany do ewaluacji na zbiorze testowym. Poniżej zakresy dostrajanych hiperparametrów i ich wytłumaczenie:

- `learning_rate`  
  - współczynnik uczenia sieci
  - rozkład: log-jednostajny  
  - zakres: 1e-4 – 3e-3

- `metadata_num_hidden_layers`  
  - liczba warstw ukrytych w MLP dla metadanych  
  - wartości: 1, 2, 3, 4, 5

- `metadata_hidden_size`  
  - liczba neuronów w warstwach ukrytych MLP dla metadanych  
  - wartości: 64, 128, 256

- `metadata_output_size`  
  - rozmiar wektora wyjściowego MLP dla metadanych
  - wartości: 32, 64, 128

- `lightcurve_num_hidden_layers`  
  - liczba warstw ukrytych w LSTM
  - wartości: 1, 2, 3, 4, 5

- `lightcurve_hidden_size  `
  - liczba neuronów w warstwach ukrytych LSTM
  - wartości: 64, 128, 256

- `classifier_hidden_size`  
  - liczba neuronów w warstwach ukrytych głowicy klasyfikacyjnej
  - wartości: 128, 256, 512

- `classifier_num_hidden_layers`  
  - liczba warstw ukrytych głowicy klasyfikacyjnej 
  - wartości: 1, 2, 3, 4, 5

- [`dropout`](https://en.wikipedia.org/wiki/Dilution_(neural_networks))
  - prawdopodobieństwo odrzucenia - w celu zapobiegania przeuczeniu
  - rozkład: jednostajny  
  - zakres: 0.0 – 0.4

- `epochs`  
  - maksymalna liczba epok treningu  
  - wartość: 100

- `early_stop_patience`  
  - liczba epok bez poprawy przed wczesnym zatrzymaniem uczenia 
  - wartość: 10

- `batch_size`  
  - liczba próbek w jednej paczce treningowej  
  - wartości: 32, 64, 128


### Hiperparametry najlepszego modelu
```python
SupernovaClassifierV1Config(
    metadata_input_size=20, 
    metadata_num_hidden_layers=5, 
    metadata_hidden_size=128, 
    metadata_output_size=128, 
    lightcurve_input_size=6, 
    lightcurve_num_hidden_layers=4, 
    lightcurve_hidden_size=64, 
    classifier_hidden_size=512, 
    classifier_num_hidden_layers=3, 
    num_classes=14, 
    dropout=0.06791426425250689
)
```

## Wyniki
Wyniki najlepszego modelu na zbiorze testowym

### Zagregowane metryki

|                   | Dokładność | Precyzja | Czułość | F1   |
|-------------------|------------|----------|---------|------|
| Mikro-uśrednianie | 0.70       | 0.70     | 0.70    | 0.70 |
| Makro-uśrednianie | 0.56       | 0.54     | 0.56    | 0.55 |

### Metryki z podziałem na klasy

| Klasa    | Precyzja | Czułość | F1   |
|----------|----------|---------|------|
| Klasa 6  | 0.46     | 0.50    | 0.48 |
| Klasa 15 | 0.63     | 0.72    | 0.67 |
| Klasa 16 | 0.88     | 0.88    | 0.88 |
| Klasa 42 | 0.45     | 0.45    | 0.45 |
| Klasa 52 | 0.00     | 0.00    | 0.00 |
| Klasa 53 | 0.00     | 0.00    | 0.00 |
| Klasa 62 | 0.53     | 0.28    | 0.37 |
| Klasa 64 | 0.39     | 0.60    | 0.47 |
| Klasa 65 | 0.88     | 0.96    | 0.92 |
| Klasa 67 | 0.00     | 0.00    | 0.00 |
| Klasa 88 | 0.96     | 0.96    | 0.96 |
| Klasa 90 | 0.68     | 0.83    | 0.75 |
| Klasa 92 | 0.86     | 0.82    | 0.84 |
| Klasa 95 | 0.90     | 0.84    | 0.87 |

### Macierz pomyłek
|        | 6  | 15 | 16  | 42 | 52 | 53 | 62 | 64 | 65  | 67 | 88 | 90  | 92 | 95 |
|--------|----|----|-----|----|----|----|----|----|-----|----|----|-----|----|----|
| **6**  | 11 | 0  | 6   | 0  | 0  | 0  | 0  | 0  | 3   | 0  | 0  | 0   | 2  | 0  |
| **15** | 0  | 49 | 0   | 6  | 0  | 0  | 1  | 0  | 0   | 0  | 0  | 12  | 0  | 0  |
| **16** | 0  | 0  | 122 | 0  | 0  | 0  | 0  | 0  | 13  | 0  | 1  | 0   | 3  | 0  |
| **42** | 2  | 11 | 0   | 82 | 0  | 0  | 12 | 4  | 0   | 0  | 0  | 70  | 0  | 2  |
| **52** | 0  | 0  | 0   | 11 | 0  | 0  | 1  | 1  | 0   | 0  | 0  | 16  | 0  | 0  |
| **53** | 6  | 0  | 0   | 0  | 0  | 0  | 0  | 0  | 0   | 0  | 0  | 0   | 0  | 0  |
| **62** | 2  | 1  | 0   | 33 | 0  | 0  | 24 | 8  | 0   | 0  | 0  | 16  | 0  | 1  |
| **64** | 1  | 0  | 0   | 2  | 0  | 0  | 2  | 9  | 1   | 0  | 0  | 0   | 0  | 0  |
| **65** | 0  | 0  | 5   | 0  | 0  | 0  | 0  | 0  | 130 | 0  | 0  | 0   | 0  | 0  |
| **67** | 0  | 1  | 0   | 7  | 0  | 0  | 4  | 0  | 0   | 0  | 0  | 19  | 0  | 0  |
| **88** | 0  | 1  | 0   | 0  | 0  | 0  | 0  | 0  | 0   | 0  | 54 | 1   | 0  | 0  |
| **90** | 2  | 12 | 0   | 41 | 0  | 0  | 1  | 1  | 0   | 0  | 0  | 281 | 0  | 0  |
| **92** | 0  | 0  | 6   | 0  | 0  | 0  | 0  | 0  | 0   | 0  | 1  | 0   | 31 | 0  |
| **95** | 0  | 3  | 0   | 1  | 0  | 0  | 0  | 0  | 0   | 0  | 0  | 1   | 0  | 27 |

### Komentarz do wyników

Rozjazd między wartościami mikro i makro nie jest zaskakujący przy niezrównoważonym zbiorze. Model dość dobrze generalizuje wiedzę na danych niewykorzystywanych podczas treningu i walidacji.

## Ograniczenia i założenia na potrzeby projektu
Na koniec chcielibyśmy wytłumaczyć jedno ograniczenie modelu w obecnej formie - podczas treningu i ewaluacji używaliśmy wyłącznie niedużej części danych treningowych z punktu widzenia konkursu, zawierających dokładne wartości spektroskopowe przesunięcia ku czerwieni, co nie jest w pełni reprezentatywne dla danych obserwacyjnych LSST i oficjalnego zbioru testowego - aby podejść do problemu tak poważnie, przydałoby się wytrenować model na pełnym zestawie danych, do czego nie mieliśmy wystarczającej mocy obliczeniowej. W praktycznych zastosowaniach astronomicznych dostępne będą głównie fotometryczne przesunięcia ku czerwieni oraz silniejsza nierównowaga klas, co może obniżyć skuteczność modelu.
