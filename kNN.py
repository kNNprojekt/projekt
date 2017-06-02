import numpy as np
import random
import os, os.path, shutil
from sklearn.datasets import load_files  # Ładowanie metody load_files, zachowującej strukturę katalogu   
from sklearn.feature_extraction.text import CountVectorizer # W celu utworzenia wektorowej reprezentacji tesktu
from sklearn.feature_extraction.text import TfidfVectorizer # W celu utworzenia wektorowej reprezentacji tesktu
from sklearn.neighbors import KNeighborsClassifier # W celu zastosowanie metody k najbliższych sąsiadów
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression # W celu utworzenia klasyfikatora, poprzez zastosowanie modelu liniowego.
from sklearn.model_selection import GridSearchCV # W celu regularyzacji regresji logistycznej
from sklearn.model_selection import train_test_split # W celu automatycznego podziału danych na część treningową i testową
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # W celu usunięcia najczęściej spotykanych słów angielskich
from sklearn.metrics import accuracy_score # Do pomiaru dokładności predykcji


class cd:
    """Menedżer kontekstowy do zmiany katalogu. 
      
           Klasa pobrana z Internetu. Żródło: StackOverflow.
           Bezpośrednie użycie metody os.chdir() bez tego rodzaju opakowania
           nie jest bezpiecznie.
   """
    def __init__(self, ścieżka):
        self.ścieżka = os.path.expanduser(ścieżka)

    def __enter__(self):
        self.inna_ścieżka = os.getcwd()
        os.chdir(self.ścieżka)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.inna_ścieżka)



class PodziałDanych:
    def dzielenie_danych(folder):
        """Metoda służy do podziału danych na treningowe i uczące

               Otwiera ona katalog 'folder' czyta jego podkatalogi, następnie tworzy dodatkowe podkatalogi
               o nazwach, które są nazwami dotychczasowych podkatalogów z dodatkiem '_train'
               oraz '_test', do których skopiuje losowe wybrane pliki z dotychczasowy katalogów. os
               Struktura katalogu 'folder' będzie zatem taka:
               - folder
               -- (dotychczasowe podkatalogi)
               -- treningowe
               --- (zawiera nazwy dotychczasowych katalogów z 80% plików)
               -- testowe
               --- (zawiera nazwy dotychczasowych katalogów z 20% plików)
        """
        mniam = os.listdir(folder)
        if "treningowe" in mniam:
            shutil.rmtree(folder + "/treningowe")
            shutil.rmtree(folder + "/testowe")
        mniam = os.listdir(folder)
        if "treningowe" in mniam:
           mniam.remove("treningowe")
        if "testowe" in mniam:
           mniam.remove("testowe")
        print(mniam)
        nowy_folder = folder + "/treningowe"
        if not os.path.exists(nowy_folder):
            os.makedirs(nowy_folder)
        nowy_folder = folder + "/testowe"
        if not os.path.exists(nowy_folder):
            os.makedirs(nowy_folder)
        tren = folder + "/treningowe"
        with cd(tren):
           for nazwa_podkatalogu in mniam:
               if not os.path.exists(nazwa_podkatalogu):
                   os.makedirs(nazwa_podkatalogu)
        test = folder + "/testowe"
        with cd(test):
           for nazwa_podkatalogu in mniam:
               if not os.path.exists(nazwa_podkatalogu):
                   os.makedirs(nazwa_podkatalogu)
        with cd(folder):
            for nazwa_katalogu in mniam:
                katalog = nazwa_katalogu
                with cd(katalog):
                    f = []
                    print("miau")
                    g = [nazwa for nazwa in os.listdir('.') if os.path.isfile(nazwa)]
                    leng = len(g)
                    print("leng/5", leng//5) 
                    ciag_wylosowany = random.sample(g, leng//5)
                    g2 = ciag_wylosowany
                    g1 = [nazwa for nazwa in g if nazwa not in g2]
                    print("Dl. g1 {}".format(len(g1)))
                    for nazwa_pliku in g1:
                        shutil.copy2(nazwa_pliku, "/home/kNN/" + 
                               folder + "/treningowe/" + katalog + "/" + nazwa_pliku)
                    for nazwa_pliku in g2:
                        shutil.copy2(nazwa_pliku, "/home/kNN/" +
                               folder + "/testowe/" + katalog + "/" + nazwa_pliku)
        print("Koniec podziału na dane treningowe i testowe")

    def ładowanie_danych(folder):
       """Metoda pobierające dane z podfolderów.
           
             Zadaniem tej metody jest załadowanie danych z podfolderów 'treningowe'
             i 'testowe' za pomocą metody load_files z modułu sklearn_datasets.
       """
       tren = "/treningowe"
       test = "/testowe"
       dane_treningowe = load_files(folder + tren)
       dane_testowe = load_files(folder + test)
       return dane_treningowe, dane_testowe
                    


    def wektorowanie(folder, metoda="kNN", min = 5, max = 0.5, słowa_do_usunięcia = False):
        """Tworzenie macierzowej reprezentacji tekstów za pomocą klasy 'CountVectorizer' i jej metod 'fit' i 'transform'
               Za pomocą klasy 'CountVectorizer' i jej metod 'fit' i 'transform' tworzona jest macierzowa 
               reprezentacja tekstu. Reprezentacja ta stosuje rzadke macierze, zdefiniowane w pakiecie SciPy
        """
        print("Początek wektorowania....")
        PodziałDanych.dzielenie_danych(folder)
        dane = PodziałDanych.ładowanie_danych(folder)
        dane_treningowe, dane_testowe = dane[0], dane[1]
        teksty_uczace, etykiety1 = dane_treningowe.data, dane_treningowe.target
        teksty_testowe, etykiety2 = dane_testowe.data, dane_testowe.target
        if metoda == "LR":
            if słowa_do_usunięcia == True:
               wektoryzator = CountVectorizer(min_df = min, max_df = max, stop_words = 'english')
                
            else:
               wektoryzator = CountVectorizer(min_df = min, max_df = max, stop_words = 'english')
            print("Utworzono wektoryzator...") 
            X_treningowe = wektoryzator.fit_transform(teksty_uczace)
            print("Utworzono X_treningowe.")
            X_testowe = wektoryzator.transform(teksty_testowe)
            print("Utworzono X_testowe.")
        elif metoda == "kNN":
            wektoryzator = TfidfVectorizer(sublinear_tf=True, min_df=min, max_df=max, stop_words='english')
            X_treningowe = wektoryzator.fit_transform(teksty_uczace)
            print("Utworzono X_treningowe.")
            X_testowe = wektoryzator.transform(teksty_testowe)
            print("Utworzono X_testowe.")
        y_treningowe = etykiety1
        y_testowe = etykiety2
        print("Koniec wektorowania")
        return X_treningowe, y_treningowe, X_testowe, y_testowe, metoda 




class KlasyfikatorTekstów():
    """Klasa, służąca do klasyfikacji danych tekstowych z pomocą regresji logistycznej oraz metody k najbliższych sąsiadów.
        
           Używając modułów, dostępnych w pakiecie scikit-learn, metody klasy KlasyfikatorTekstów dokonują
           klasyfikacji danych tekstowych, zawartych w katalogu 'folder'. Zakłada się, że katalog 'folder'
           zawiera podkatalogi, których nazwy są zarazem etykietami kategorii, do których klasyfikowane
           będą dane testowe.
    """
    def __init__(self, folder, metoda):
        """Inicjalizator (pseudo-konstruktor) pobiera dane z katalogu 'folder' organizując je na potrzeby innych metod"""

        Xym = PodziałDanych.wektorowanie(folder, metoda)
        self.X_tren, self.y_tren, self.X_test, self.y_test = Xym[0], Xym[1], Xym[2], Xym[3]  
        self.metoda = Xym[4]    
   
        # Wyświetlane podstawowych informacji na temat typów danych, podlegających klasyfikacji.
        # kształtów ('shape') tablic oraz wielkości katalogów z danymi treningowymi i testowymi.

        print("Typ danych X_tren {} ".format(type(self.X_tren)))
        print("Kształt tablicy X_tren {} ".format(self.X_tren.shape))
        print("Typ danych X_test {} ".format(type(self.X_test)))
        print("Kształt tablicy X_test {} ".format(self.X_test.shape))
        print("Typ danych y_tren {} ".format(type(self.y_tren)))
        print("Kształt tablicy y_tren {} ".format(self.y_tren.shape))
        print("Typ danych y_test {} ".format(type(self.y_test)))
        print("Kształt tablicy y_testowe {} ".format(self.y_test.shape))
        print("Liczba plików w poszczególnych podfolderach: {} ".format(np.bincount(self.y_tren)))
        print("Liczba plików w poszczególnych podfolderach: {} ".format(np.bincount(self.y_test)))
        print("Pierwsze X_tren {0}, {1}, {2} ".format(self.X_tren[0, 0], self.X_tren[0, 1], self.X_tren[0, 2]))
        print("Pierwsze X_test {0}, {1}, {2} ".format(self.X_test[0, 0], self.X_test[0, 1], self.X_test[0, 2]))      


    def tworzenie_klasyfikatora(self, liczba_sąsiadów = 1):
        """Tworzenie klasyfikatora za pomocą modelu liniowego regresji logistycznej i ocena jego dokładności"""
        print("Początek tworzenia klasyfikatora; tworzenie_klasyfikatora: wywołanie metody wektorowanie...")
        if self.metoda == "LR":
            lr = LogisticRegression()
            lr.fit_transform(self.X_tren, self.y_tren)
#            print("Długości list przewidywane oraz y_test: {0}, {1} ".format(len(przewidywane), len(self.y_test)))
            try:
                wynik_cross = cross_val_score(LogisticRegression(), self.X_tren, self.y_tren, cv = 5)
                przewidywane = lr.predict(self.X_test)
                print("tworzenie_klasyfikatora(LR) - w 'try'")
                wynik1 = accuracy_score(self.y_test, przewidywane)
                print("Wynik 1", wynik1)
                uff = lr.score(self.X_test, self.y_test)
                print("uff", uff)
            except(ValueError):
                print("tworzenie_klasyfikatora(LR) - w 'except'")
                maks0 = np.min([self.X_tren.shape[0], self.X_test.shape[0]])
                maks1 = np.min([self.X_tren.shape[1], self.X_test.shape[1]])
                print("maks: {0}, {1} ".format(maks0, maks1))
                X_tr = self.X_tren[:maks0, :maks1]
                X_te = self.X_test[:maks0, :maks1]
                y_tr = self.y_tren[:maks0]
                y_te = self.y_test[:maks0]
                lr.fit(X_tr, y_tr)
                przewidywane = lr.predict(X_te)
                wynik = accuracy_score(y_te, przewidywane)
                print("Wynik", wynik)
                uff = lr.score(X_te, y_te)
                print("uff", uff)
            finally:
                print("Wynik na tekstach testowych: {} ".format(uff))
                wynik = uff
        elif self.metoda == "kNN":
            knn = KNeighborsClassifier(liczba_sąsiadów)
            knn.fit(self.X_tren, self.y_tren)
            try:
                przewidywane = knn.predict(self.X_test)
                print("tworzenie_klasyfikatora(kNN) - w 'try'")
                wynik1 = accuracy_score(self.y_test, przewidywane)
                print("Wynik 1", wynik1)
                uff = knn.score(self.X_test, self.y_test)
                print("uff", uff)
            except(ValueError):
                print("tworzenie_klasyfikatora(kNN) - w 'except'")
                maks0 = np.min([self.X_tren.shape[0], self.X_test.shape[0]])
                maks1 = np.min([self.X_tren.shape[1], self.X_test.shape[1]])
                print("maks: {0}, {1} ".format(maks0, maks1))
                X_tr = self.X_tren[:maks0, :maks1]
                X_te = self.X_test[:maks0, :maks1]
                y_tr = self.y_tren[:maks0]
                y_te = self.y_test[:maks0]
                knn.fit(X_tr, y_tr)
                przewidywane = knn.predict(X_te)
                wynik = accuracy_score(y_te, przewidywane)
                print("Wynik", wynik)
                uff = knn.score(X_te, y_te)
            finally:
                print("Wynik na tekstach testowych: {} ".format(uff))
                wynik = uff
        print("Koniec tworzenia klasyfikatora, {} ".format(wynik))
        return wynik


    def regularyzacja_klasyfikatora(self, *args):
        """Regularyzacja klasyfikatora za pomocą parametru C klasy LogisticRegression"""
        liczba_argumentów = len(args)
        if liczba_argumentów == 0:
            X_treningowe = self.wektorowanie()[0]       
            X_testowe = self.wektorowanie()[3]         
        elif liczba_argumentów == 1:
            X_treningowe = self.wektorowanie(args[0])[0]
            X_testowe = self.wektorowanie(args[0])[3]
        elif liczba_argumentów == 2:
            X_treningowe = self.wektorowanie(args[0], args[1])[0]
            X_testowe = self.wektorowanie(args[0], args[1])[3]
        elif liczba_argumentów == 3:
            X_treningowe = self.wektorowanie(args[0], args[1], args[2])[0]
            X_testowe = self.wektorowanie(args[0], args[1], args[2])[3]
        parametryC = {'C': [0.001, 0.01, 0.1, 1, 10]}
        siatka = GridSearchCV(LogisticRegression(), parametryC, cv = 5)
        y_treningowe = self.etykiety1
        y_testowe = self.etykiety1
        siatka.fit(X_treningowe, y_treningowe)
        najlepszy_wynik = siatka.best_score_
        najlepsze_parametry = siatka.best_params_
#        wynik_dla_danych_testowych = siatka.score(X_testowe, y_testowe)
        return najlepszy_wynik, najlepsze_parametry

     

    def wyświetlanie_informacji(self):
        """ Wyświetlanie informacji na temat tekstów, podlegających klasyfikacji i ich reprezentacji macierzowej """
                
        # Nie od rzeczy będzie wyświetlenie informacji na temat typu i wielkości zmiennej 'macierzowa_reprezentacja_tekstu'
        print()
        wek = self.wektorowanie()[2]
        nazwy_cech = wek.get_feature_names()
        print("Liczba cech: {}".format(len(nazwy_cech)))
        print("Pierwszych 10 cech: {}".format(nazwy_cech[:10]))

        # Tutaj wyświetlimy średnią dokładność, osiąganą w modelu liniowym poprzez użycie regresji logistycznej
        rezultat = self.tworzenie_klasyfikatora()
        print("Średnia dokładność modelu: {:3f}".format(np.mean(rezultat)))

        # Tutaj pokazujemy, czy regularyzacja poprawia wyniki.
        rezultat = self.regularyzacja_klasyfikatora()
        print ("Średnia dokładność po regularyzacji: {:3f}".format(rezultat[0]))
        print ("Najlepszy parametr regularyzacji: ", rezultat[1])

        # Teraz wyświetlamy zmiany po wprowadzeniu ograniczeń na występowanie ciągów znaków w dokumentach.
        rezultat = self.klasyfikowanie_z_ograniczeniem_występowania(3, 50)
        print("Średnia dokładność modelu z ograniczeniem \nna występowanie ciągów znaków: {:3f}".format(np.mean(rezultat)))
        

        # Teraz wyświetlamy zmiany po wprowadzeniu ograniczeń na występowanie ciągów znaków w dokumentach,
        # a także bez uwzględniania 318 najczęściej występujących słów w języku angielskim
        rezultat = self.klasyfikowanie_z_ograniczeniem_występowania(3, 50, True)
        print("Średnia dokładność modelu z ograniczeniem na występowanie ciągów znaków" +  
              "\ni bez uwzględniania najczęściej występujących słow: {:3f}".format(np.mean(rezultat)))

        # Wynikiem kolejnego testu jest wyświetlenie dokładności w przypadku, gdy wprowadzamy ograniczenia,
        # usuwamy często występujące słowa, i manipulujemy parametrem 'C'.
        rezultat = self.regularyzacja_klasyfikatora(3, 50, True)
        print ("Średnia dokładność po regularyzacji, z ograniczeniem na występowanie ciągów znaków" +  
              "\ni bez uwzględniania najczęściej występujących słow : {:3f}".format(rezultat[0]))
        print ("Najlepszy parametr regularyzacji: ", rezultat[1])






class Klasyfikator_Tekstów_RęcznyKNN(PodziałDanych):
    """Klasa, służąca do klasyfikacji danych tekstowych z pomocą metody k najbliższych sąsiadów.
        
           NIE używając modułów, dostępnych w pakiecie scikit-learn, metody klasy KlasyfikatorTekstów dokonują
           klasyfikacji danych tekstowych, zawartych w katalogu 'folder'. Zakłada się, że katalog 'folder'
           zawiera podkatalogi, których nazwy są zarazem etykietami kategorii, do których klasyfikowane
           będą dane testowe.
    """
    def __init__(self, folder):
        """Inicjalizator (pseudo-konstruktor) pobiera dane z katalogu 'folder' organizując je na potrzeby innych metod"""
        tren = "/treningowe"
        test = "/testowe"
        PodziałDanych.dzielenie_danych("text_files")


 
    


def main(): 
    print("main: tworzenie obiektu KlasyfikatorTekstów...")
    klasyfikator_tekstów = KlasyfikatorTekstów("text_files", "LR")
    print("main: koniec konstruktora KlasyfikatorTekstów, wynik:")
    wynik = klasyfikator_tekstów.tworzenie_klasyfikatora(10)


if __name__ == "__main__":
    main()
        

