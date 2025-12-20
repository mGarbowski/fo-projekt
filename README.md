# Klasyfikacja typu supernowy na podstawie danych fotometrycznych

Projekt semestralny na przedmiot Fizyka Ogólna (2025Z).

## Autorzy

* Maksym Bieńkowski
* Mikołaj Garbowski

## Opis

W dostępnym zbiorze danych jest niewielki zbiór treningowy z etykietami i ogromny zbiór testowy bez etykiek.
Posłużymy się zbiorem treningowym, podzielimy go na zbiór treningowy i walidacyjny.
Wytrenujemy klasyfikator i będziemy oceniać dokładność klasyfikacji na zbiorze walidacyjnym.
Będziemy wyznaczać dokładność przy mikro-uśrednianiu i makro-uśrednianiu.

Istnieje też zbiór testowy z udostępnionymi etykietami.
Jak starczy czasu to ocenimy na nim nasz model (ale to zobaczymy, bo zbiór jest ogromny).

### Pomysły na rozwiązanie

Testujemy podejścia znane z zajęć i w granicach możliwości co do zasobów obliczeniowych (nie trenujemy transformera).

#### Architektura modelu
* Model oparty na sieciach rekurencyjnych (LSTM)
  * po 1 sieci na każde z 6 pasem
  * połączenie wyjść sieci dla każdego pasma, metadanych i wyznaczonych cech
### Inżynieria cech
* Do przeanalizowania które atrybuty z metadanych bierzemy do modelu
* Wyznaczenie cech per szereg czasowy
  * średnia
  * odchylenie
  * jakieś inne, może maksymalna częstotliwość z FFT

## Źródła
* [Konkurs PLAsTiCC na Kaggle (ze zbiorem danych)](https://www.kaggle.com/competitions/PLAsTiCC-2018/data)
* [Opis zwycięskiego rozwiązania z konkursu](https://www.kaggle.com/competitions/PLAsTiCC-2018/writeups/kyle-boone-overview-of-1st-place-solution)
* [Notatniki udostępnione przez autorów zbioru](https://github.com/lsstdesc/plasticc-kit)
