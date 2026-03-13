import numpy as np
import tensorflow as tf
import time



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



def tarkkuus(w1, bias1, w2, bias2, x_testi, y_testi, maara):
    '''
    Laskee neuroverkon onnistumisprosentin.
    
    Parametrit:
        w1, w2, bias1, bias2: Painot ja biasit
        x_testi: Testikuvat
        y_testi: Testikuvien oikeat vastaukset
        maara: Testattavien kuvien määrä
    
    Palauttaa:
        onnistumis_prosentti: tarkkuus float tyyppisenä
    '''
    if maara is None:
        maara = y_testi.size
    
    testi_kuvat = x_testi[:, :maara]
    vastaukset = y_testi[:maara]

    _, _, _, oletus = kytkos(w1, bias1, w2, bias2, testi_kuvat)

    oletukset = np.argmax(oletus, axis=0)

    onnistuneet = np.sum(oletukset == vastaukset)

    onnistumis_prosentti = onnistuneet / maara * 100

    return onnistumis_prosentti



if __name__ == "__main__":
    '''
    Pääohjelma, joka toimii valmiiksi treenautetulla .npz tiedostolla

    Laskee verkon tarkkuuden.
    '''
    tiedosto = 'painot3.npz'
    testien_maara = 10000

    try:
        parametrit = np.load(tiedosto)

        w1 = parametrit['w1']
        bias1 = parametrit['bias1']
        w2 = parametrit['w2']
        bias2 = parametrit['bias2']

        print("Tiedosto luettu onnistuneesti")

        _, _, x_testi, y_testi = mnist_lataus()

        alkuaika = time.perf_counter()

        tarkkuus_prosentti = tarkkuus(w1, bias1, w2, bias2, x_testi, y_testi, testien_maara)

        loppuaika = time.perf_counter()

        kulunut_aika = loppuaika - alkuaika
        kulunut_aika_ms = kulunut_aika * 1000

        print(f"Verkon onnistumisprosentti on {tarkkuus_prosentti:.2f}%")
        print(f"Laskennan kulunut aika oli {kulunut_aika_ms:.2f} ms")

    except FileNotFoundError:
        print("Tiedostoa ei löytynyt")

    except KeyError as ke:
        print("Tiedostosta puuttuu parametri " + ke)
