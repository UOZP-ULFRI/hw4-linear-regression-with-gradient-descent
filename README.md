# Implementacija linearne regresije z gradientnim sestopom

V nalogi boste implementirati linearno regresijo s pomočjo gradientnega sestopa.


Predlogo hw4.py dopolnite z implementacijami metod `cost`, `gradient`, `gradient_descent`,
`LinearRegression.fit` in `LinearRegression.predict`. Vhodi in izhodi metod so dodatno 
razloženi v komentarjih posamezne metode. Omenjene metode implementirajte z učinkovitimi 
matrično-vektorsko operacijami. Na to vas bodo prijazno opozorili tudi testi.

Metode nato uporabite še na [priloženih podatkih](./data.tab). Pri tem si pomagajte z
metodo `file_reader`, ki jo implementirate sami. Narišite in shranite dve sliki, ki predstavljata
graf s premico v prostoru dveh izbranih spremenljivk. Prva naj bo za spremenljivko z največjim druga pa z 
najmanjšim vplivom na ciljno spremenljivko. Kodo pišite pod vrstico `if __name__ == "__main__":`. 

Ko boste vizualizaciji pravilno shranili, se bosta izrisali spodaj. 
Shranite ju pod imenom `image1.png` ter `image2.png`.

# Oddaja

Nikakor ne spreminjate datoteke s testi. Na koncu oddajte dopolnjeno datoteko hw4.py, 
ki prestane vse teste. Preberite opis in se držite tipov, ki jih metoda prejme in vrača. 
Za testiranje metod smo dodali še datoteko test_hw4.py.

Pri implementaciji metod si pomagajte le s knjižnico `Numpy`! 
Za vizualizacijo pa uporabite knjižnico `matplotlib`.



<div>
    <img src="image1.png" width="500">
</div>


<div>
    <img src="image2.png" width="500">
</div>