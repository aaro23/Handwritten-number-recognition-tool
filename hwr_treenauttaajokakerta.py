import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps


def mnist_lataus():
    '''
    Lataa ja esikäsittelee MNIST-datan.
    
    Esikäsittelyssä 28x28 kuvista tehdään yksittäisiä 784 alkion vektoreita.
    Vektoreiden pikseliarvot skaalataan välille 0-1 ja vektorit transponoidaan ns. pystyyn laskutoimituksia varten
    
    Funktio palauttaa:
        x_treeni: opetusdatan pikselit muodossa (784, 60000)
        y_treeni: opetusdatan oikeat vastaukset muodossa (60000, )
        x_testi: testidatan pikselit muodossa (784, 10000)
        y_testi: testidatan oikeat vastaukset muodossa (10000, )
    '''
    (x_treeni_alku, y_treeni_alku), (x_testi_alku, y_testi_alku) = tf.keras.datasets.mnist.load_data()

    # Muutetaa pikseli rivit yheks pitkäks riviks
    x_treeni_lattea = x_treeni_alku.reshape(x_treeni_alku.shape[0], -1) / 255.0
    x_testi_lattea = x_testi_alku.reshape(x_testi_alku.shape[0], -1) / 255.0

    # Transponoidaan eli vaihetaan matriisin rivit ja sarakkeet päikseen
    x_treeni = x_treeni_lattea.T
    x_testi = x_testi_lattea.T

    y_treeni = y_treeni_alku
    y_testi = y_testi_alku

    return x_treeni, y_treeni, x_testi, y_testi



def parametrien_alustus():
    '''
    Alustaa neuroverkon painot ja biasit satunnaisluvuilla väliltä -0,5 ja 0,5.
    
    Palauttaa:
        w1: 1. kerroksen painot moudossa (128, 784)
        bias1: 1. kerrotken biasit muodossa (128, 1)
        w2: 2. kerroksen painot muodossa (10, 128)
        bias2: 2. kerroksen biasit muodossa (10, 1)
    '''    
    # 1. kerros
    w1 = np.random.rand(128, 784) - 0.5
    bias1 = np.random.rand(128, 1) - 0.5

    # 2. kerros
    w2 = np.random.rand(10, 128) - 0.5
    bias2 = np.random.rand(10, 1) -0.5

    return w1, bias1, w2, bias2



def relu(z):
    '''
    Muuttaa negatiiviset arvot nollaksi ja pitää positiiviset samoina.
    '''
    # Jotta saadaan kaikki negatiiviset arvot nollaksi
    return np.maximum(z, 0)



def softmax(z):
    '''
    Muuttaa neuronien arvot todennäköisyyksiksi.
    '''
    # Muuttaa output kerroksen neuronien painot prosenteiksi (eli todennäköisyyksiksi)
    a = np.exp(z)
    return a/np.sum(a, axis=0, keepdims=True)



def kytkos(w1, bias1, w2, bias2, syote):
    '''
    Yhdistää neuroverkon kerrokset ns eteenpäin.
    
    Palauttaa:
        l1: 1. kerroksen alkuperäinen tulos
        a1: 1. kerroksen ReLU tulos
        l2: 2. kerroksen elkuperäinen tulos
        a2: 2. kerroksen ulostulo (softmax)
    '''
    # 1. Kerros
    l1 = np.dot(w1, syote) + bias1
    a1 = relu(l1)

    # 2. Kerros
    l2 = np.dot(w2, a1) + bias2
    a2 = softmax(l2) # output kerroksen tulos eli oletettu numero

    return l1, a1, l2, a2



def vektori(oikea):
    '''
    Muuttaa datan vastaukset one-hot vektoreiksi.
    
    Esim. luku 2 muuttuu muotoon [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    Parametrit:
        oikea: datan antama numero väliltä 0-9
        
    Palauttaa:
        oikea_vektori.T: one-hot matriisi transponoituna muotoon (10, m)
    '''
    # Halutaan MNIST -datan antama oikea vastaus vektorimuotoon [0,0,0,0,0,0,0,0,0,0]. Esim 3 näyttäisi tältä: [0,0,0,1,0,0,0,0,0,0]
    oikea_vektori = np.zeros((oikea.size, oikea.max() + 1)) # vektori [0,0,0,0,0,0,0,0,0,0]
    
    # muutetaan oikea vastaus 1:ksi
    oikea_vektori[np.arange(oikea.size), oikea] = 1
    
    # Transponoidaan muodostettu vektori, jotta se on samansuuntainen verkon kanssa
    return oikea_vektori.T



def onko_aktivoitunut(z):
    '''
    Tarkastelee, onko jokin neuroni aktivoitunut (palauttaa 1, jos on ja 0, jos ei).

    Halutaan tietää, koska ei-aktivoituneen neuronin painoja ei haluta muokata.
    '''
    return z > 0



def takaperin(l1, a1, l2, a2, w1, w2, syote, oikea):
    '''
    Määritellään painojen ja biasien virheiden suuruus molemmissa kerroksissa käymällä neuroverkko takaperin läpi.
    
    Toisin sanoen funktio selvittää, paljonko kunkin neutronin paino ja kerroksen bias vaikuttivat virheeseen,
    jotta niitä voidaan muuttaa oikeaan suuntaan.
    
    Parametrit:
        l1: 1. kerroksen arvot
        a1: 1. kerroksen aktivoituneet arvot
        l2: 2. kerroksen arvot
        a2: 2. kerroksen antamat ulostulot (lopulliset todennäköisyydet)
        syote: treenidatan kuvat
        oikea: treenidatan oikeat vastaukset
    
        Palauttaa:
            vaikutus_w1, vaikutus_bias1: 1. kerroksen korjausarvot
            vaikutus_w2, vaikutus_bias2: 2. kerroksen korjausarvot
    '''
    kuva_maara = oikea.size
    oikea_vektori = vektori(oikea)

    # 2. Kerros (ulostulo)
    # Tutkitaan, paljonko verkon antama vastaus eroaa MNIST -datan määrittämästä oikeasta (eli virheen suuruus)
    virhe_l2 = a2 - oikea_vektori

    # Ratkaistaan, paljonko kerroksen 2 neuroneihin johtavat painot vaikuttivat tulleeseen virheeseen
    vaikutus_w2 = 1 / kuva_maara * np.dot(virhe_l2, a1.T)

    # Bias on kerroksen jokaiselle neuronille sama, joten sen virhe on kaikkien virheiden keskiarvo
    vaikutus_bias2 = 1 / kuva_maara * np.sum(virhe_l2, axis=1, keepdims=True)

    # 1. Kerros
    # Tutkitaan 1. kerroksen neuronien vaikutusta 2. kerroksen eli ulostulon neuroneihin
    virhe_l1 = np.dot(w2.T, virhe_l2) * onko_aktivoitunut(l1) # alkuosa laskee kaikille neuroneille virheen, "onko_aktivoitunut" nollaa sen neuroneilta, joita verkko ei käyttänyt

    # Ratkaistaan, paljonko syötteen eli kuvan pikseleiden painot vaikuttivat 1. kerrokseen tulleeseen virheeseen
    vaikutus_w1 = 1 / kuva_maara * np.dot(virhe_l1, syote.T)

    vaikutus_bias1 = 1 / kuva_maara * np.sum(virhe_l1, axis=1, keepdims=True)

    # Palautetaan matriisit, jotka kertovat missä neuroneissa arvoja muutetaan ja minkä verran
    return vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2



def uudet_parametrit(w1, bias1, w2, bias2, vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2, askel):
    '''
    Vaihtaa neuronien painot uusiin laskettunen virheiden perusteella.

    (Englanniksi "gradient descent")
    
    Parametrit:
        w1, bias1, w2, bias2: neuroverkon nykyiset painot ja biasit
        vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2: Takaperin-funktion antamat korjausarvot
        askel: käyttäjän (tai siis koodarin) määrittämä arvo; määrittää korjeusten suuruuden
    
    Palauttaa:
        Uudet painot ja biasit, joiden myötä pitäisi tulla jonkin verran tarkempia tuloksia
    '''
    w1 = w1 - vaikutus_w1 * askel
    bias1 = bias1 - vaikutus_bias1 * askel

    w2 = w2 - vaikutus_w2 * askel
    bias2 = bias2 - vaikutus_bias2 * askel

    return w1, bias1, w2, bias2



def treenautus_silmukka(syote, oikea, askel, toistot):
    '''
    Kokoaa neuroverkon opetukseen käytetyt funkiot samaan silmukkaan.



    Parametrit:
        syote: treenidatan kuvien pikselit
        oikea: treenidatan oikeat vastaukset
        askel: määrittää korjausten suuruuden
        toistot: montako kertaa treenautuskoodi käydään läpi
    
    Palauttaa:
        Treenautetut painot ja biasit, joita käytetään uusien kuvien tunnistamiseen
    '''
    # alustetaan painot ja biasit
    w1, bias1, w2, bias2 = parametrien_alustus()

    for i in range(toistot):
        l1, a1, l2, a2 = kytkos(w1, bias1, w2, bias2, syote) # verkko arvaa

        vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2 = takaperin(l1, a1, l2, a2, w1, w2, syote, oikea)

        w1, bias1, w2, bias2 = uudet_parametrit(w1, bias1, w2, bias2, vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2, askel)

    # Palautetaan korjaukset
    return w1, bias1, w2, bias2



class kayttoliittyma:
    '''
    Ohjelman käyttöliittymään liittyvät koodit
    '''
    def __init__(self, w1, bias1, w2, bias2):
        '''
        Käyttäjälle näkyvä osuus.
        
        Luo visuaaliset osat käyttöliittymästä eli napit ja piirtoalueen,
        mutta ei niiden toimintoja.
        '''
        self.w1 = w1
        self.bias1 = bias1
        self.w2 = w2
        self.bias2 = bias2

        self.paaikkuna = tk.Tk()
        self.paaikkuna.title("Piirrä numero väliltä 0-9")

        self.piirtoalue_leveys = 280
        self.piirtoalue_korkeus = 280
        self.kynan_koko = 12

        self.piirtoalue = tk.Canvas(self.paaikkuna, width=self.piirtoalue_leveys, height=self.piirtoalue_korkeus, bg='white', cursor="cross")
        self.piirtoalue.pack(pady=10)
        self.piirtoalue.bind("<B1-Motion>", self.piirra)

        self.kuva = Image.new("L", (self.piirtoalue_leveys, self.piirtoalue_korkeus), color=255)
        self.piilo_piirros = ImageDraw.Draw(self.kuva)

        self.nappi_tunnista = tk.Button(self.paaikkuna, text="Tunnista numero", font=("Arial", 14), command=self.arvaa)
        self.nappi_tunnista.pack(pady=5)

        self.nappi_tyhjenna = tk.Button(self.paaikkuna, text="Tyhjennä", font=("Arial", 12), command=self.tyhjenna)
        self.nappi_tyhjenna.pack(pady=5)

        self.label_tulos = tk.Label(self.paaikkuna, text="Piirrä numero ja paina Tunnista", font=("Arial", 16))
        self.label_tulos.pack(pady=10)

        self.paaikkuna.mainloop()
    


    def piirra(self, sijainti):
        '''
        Piirtoalueen toiminnalliseksi tekevä koodi.
        Eli että kynä toimii ja piirtää kuvan.
        '''
        x1, y1 = (sijainti.x - self.kynan_koko), (sijainti.y - self.kynan_koko)
        x2, y2 = (sijainti.x + self.kynan_koko), (sijainti.y + self.kynan_koko)

        self.piirtoalue.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.piilo_piirros.ellipse([x1, y1, x2, y2], fill=0)
    


    def tyhjenna(self):
        '''
        Tyhjennä napin toiminnot.
        
        Tyhjentää piirtoalueen, minkä jälkeen siihen voi piirtää uuden numeron.
        '''
        self.piirtoalue.delete("all")
        self.kuva = Image.new("L", (self.piirtoalue_leveys, self.piirtoalue_korkeus), color=255)
        self.piilo_piirros = ImageDraw.Draw(self.kuva)
        self.label_tulos.config(text="Piirrä numero ja paina Tunnista")
    


    def arvaa(self):
        '''
        Skaalaa ja keskittää piirretyn kuvan vastaamaan ohjelman
        treenidataa sekä lopussa kutsuu kytkos-funktiota saadakseen
        verkon antaman vastauksen.

        Kirjoittaa käyttäjälle nähtäville, mitä verkko on tunnistanut
        ja kuinka suurella varmuudella.

        Kuvan käsittelyn olisi ehkä voinut sijoittaa omaan funktioonsa,
        mutta koodi toimii nyt hyvin, joten annan asian olla.
        '''
        pieni_kuva = self.kuva.resize((28, 28), resample=Image.Resampling.LANCZOS)

        kaannetty_kuva = ImageOps.invert(pieni_kuva)

        pikselit = np.array(kaannetty_kuva) / 255.0

        syote = pikselit.reshape(-1, 1)

        _, _, _, oletus = kytkos(self.w1, self.bias1, self.w2, self.bias2, syote)
        tunnistanut = np.argmax(oletus)
        todennakoisyys = np.max(oletus) * 100

        self.label_tulos.config(text=f"Verkko tunnisti {tunnistanut} varmuudella {todennakoisyys}%")



if __name__ == "__main__":
    '''
    Pääohjelma.
    
    1. Lataa ja esikäsittelee datan
    2. Treenauttaa neuroverkon treenidatalla
    3. Avaa graafisen käyttöliittymän, minkä jälkeen käyttäjä voi käyttää ohjelmaa
    '''
    x_treeni, y_treeni, x_testi, y_testi = mnist_lataus()

    print("\nTreenautetaan")

    w1, bias1, w2, bias2 = treenautus_silmukka(x_treeni, y_treeni, 0.1, 200)

    print("Treenautus done")

    app = kayttoliittyma(w1, bias1, w2, bias2)

