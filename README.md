**Projekt mający na celu badanie zachowania sieci syjamskich.** 

* AGH Machine Learning 2 

Cele projektu

1. Wyuczenie sieci dla dwóch zbiorów EMNIST i CIFAR10. Dokonanie treningu na tych zbiorach dwóch sieci i określenie ich dokładność wraz z wyliczonym odchyleniem standardowym. 
Nauczenie ich dla 2 różnych inicjalizacji wag. Mają to być sieci typu vanilla networks. 

1. Następnie dla każdego z tych zbiorów: 

   2. Stworzyć podwójną sieć (Siamese), zainicjalizować ją od początku i połączyć ostatnią warstwą przed softmax’em.  

   2. Taką podwójną sieć uczyć w ten sposób by na wejście jednej i drugiej podawane były RÓŻNE przykłady z TEJ SAMEJ klasy.  

   2. Porównać acc (F1, prec/recall) dla danych testowych dla pojedynczo uczonej sieci VGG, Siamese NN, oraz dwóch części ROZCIĘTEJ sieci syjamskiej.    

   2. Wyciągnąć wnioski. 

3. Zrobić to dla zmniejszającego się zbioru wejściowego

4. Czy sieci syjamskie są bardziej odporne na zmniejszanie się zbioru danych od pojedynczych sieci? 

5. Jaki wynik otrzymamy, gdy sieci syjamskie będziemy uczyć także dla danych wejściowych pochodzących zarówno z RÓŻNYCH i TYCH samych klas, wprowadzając dodatkowy neuron wyjściowy (binarny pokazujący czy przykłady są z tej czy różnych klas). 

Wyniki projektu w pliku [readme.pdf](readme.pdf)
