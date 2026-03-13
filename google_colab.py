import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    '''
    # kommentoitu pois
    print("prekäsittely valmis")
    print(f"x_treeni muoto nyt: {x_treeni.shape}")
    print(f"y_treeni muoto nyt: {y_treeni.shape}")
    '''
    return x_treeni, y_treeni, x_testi, y_testi



def parametrien_alustus():
    '''
    Alustaa neuronien painot normaalijakaumalla ja asettaa biasit nolliksi.

    Palauttaa:
        w1: 1. kerroksen painot moudossa (128, 784)
        bias1: 1. kerrotken biasit muodossa (128, 1)
        w2: 2. kerroksen painot muodossa (10, 128)
        bias2: 2. kerroksen biasit muodossa (10, 1)
    '''
    # 1. kerros
    w1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
    bias1 = np.zeros((128, 1))

    # 2. kerros
    w2 = np.random.randn(10, 128) * np.sqrt(2.0 / 128)
    bias2 = np.zeros((10, 1))

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
    z2 = z - np.max(z, axis=0, keepdims=True) # jotta arvot eivät ryöpsähdä käsiin
    a = np.exp(z2)
    return a/np.sum(a, axis=0, keepdims=True)



def kytkos(w1, bias1, w2, bias2, syote):
    '''
    Yhdistää neuroverkon kerrokset ns eteenpäin.

    Palauttaa:
        l1: 1. kerroksen alkuperäinen tulos
        a1: 1. kerroksen ReLU tulos
        l2: 2. kerroksen elkuperäinen tulos
        a2: 2. kerroksen ulostulo (softmax)'''
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



def treenautus_silmukka(syote, oikea, alku_askel, toistot, batch_koko=256):
    '''
    Säätää painoja ja biaseja uudelleen ja uudelleen.

    Parametrit:
        syote: treenikuvat
        oikea: treenikuvien oikeat vastaukset
        alku_askel: oppimisnopeus aluksi
        toistot: montako kertaa opetusdata käydään läpi
        batch_koko: kerralla käsiteltävien kuvien määrä
    
    Palauttaa:
        w1, bias1, w2, bias2: uudet painot ja biasit
        toisto_historia: lista toistokerroista kuvaajaa varten
        tarkkuus_historia: lista tarkkuuden kehittymisestä kuvaajaa varten
    '''
    # alustetaan painot ja biasit
    w1, bias1, w2, bias2 = parametrien_alustus()

    kuva_maara = syote.shape[1]


    toisto_historia = []
    tarkkuus_historia = []



    for i in tqdm(range(toistot), desc="Treenautus käynnissä"):

        askel = alku_askel * (0.5 ** (i / (toistot / 2)))

        # Sekoitetaan data jokaisella kierroksella (estää ulkoa opettelun)
        sekoitus_indeksit = np.random.permutation(kuva_maara)
        sekoitettu_syote = syote[:, sekoitus_indeksit]
        sekoitettu_oikea = oikea[sekoitus_indeksit]

        # Käydään data läpi pienissä paloissa
        for j in range(0, kuva_maara, batch_koko):
            # Erotetaan batchin kokoinen siivu datasta
            syote_batch = sekoitettu_syote[:, j:j+batch_koko]
            oikea_batch = sekoitettu_oikea[j:j+batch_koko]

            l1, a1, l2, a2 = kytkos(w1, bias1, w2, bias2, syote_batch)

            # Lasketaan virhe ja tarvittavat korjaukset batchille
            vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2 = takaperin(l1, a1, l2, a2, w1, w2, syote_batch, oikea_batch)

            # Päivitetään painot
            w1, bias1, w2, bias2 = uudet_parametrit(w1, bias1, w2, bias2, vaikutus_w1, vaikutus_bias1, vaikutus_w2, vaikutus_bias2, askel)

        if i % 10 == 0 or i == toistot - 1:
            _, _, _, a2 = kytkos(w1, bias1, w2, bias2, syote)
            ennustukset = np.argmax(a2, axis=0)
            tarkkuus = np.sum(ennustukset == oikea) / oikea.size
            print(f"Toistot: {i:3d} | Askel: {askel:.4f} | Treenitarkkuus: {tarkkuus * 100:.2f}%")

            toisto_historia.append(i)
            tarkkuus_historia.append(tarkkuus * 100)

    return w1, bias1, w2, bias2, toisto_historia, tarkkuus_historia



if __name__ == "__main__":
    '''
    Pääohjelma:

    Lataa ja esikäsittelee datan

    Treenauttaa neuroverkkoa treenidatalla
    
    Tallentaa uudet painot ja biasit tiedostoon
    
    Piirtää kuvaajan verkon oppimisesta, ja testaa sekä tulostaa lopullisen tarkkuuden
    '''
    x_treeni, y_treeni, x_testi, y_testi = mnist_lataus()

    print("\nTreenautetaan")

    w1, bias1, w2, bias2, toistot, tarkkuudet = treenautus_silmukka(x_treeni, y_treeni, 0.1, 150)

    _, _, _, a2 = kytkos(w1, bias1, w2, bias2, x_testi)
    ennustukset = np.argmax(a2, axis=0)
    tarkkuus = np.sum(ennustukset == y_testi) / y_testi.size
    print("Treenautus done")
    print(f"Testitarkkuus: {tarkkuus * 100:.2f}%")

    np.savez('painot3.npz', w1=w1, bias1=bias1, w2=w2, bias2=bias2)
    print("Painot tallennettu")

    plt.figure(figsize=(8, 5))
    plt.plot(toistot, tarkkuudet, marker='o', color='blue', label='Treenitarkkuus')
    plt.title('Neuroverkon oppimisen edistyminen')
    plt.xlabel('Toistot')
    plt.ylabel('Treenitarkkuus')
    plt.legend()
    plt.grid(True)
    plt.show()
