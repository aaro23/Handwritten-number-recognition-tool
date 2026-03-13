import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps


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
        self.kynan_koko = 25

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
        kaannetty_kuva = ImageOps.invert(self.kuva)

        aariviivat = kaannetty_kuva.getbbox()

        if aariviivat is None:
            self.label_tulos.config(text="Et piirtänyt :(")
            return
        
        leikattu_kuva = kaannetty_kuva.crop(aariviivat)

        leveys, korkeus = leikattu_kuva.size
        kerroin = 20.0 / max(leveys, korkeus)
        uus_leveys = int(leveys * kerroin)
        uus_korkeus = int(korkeus * kerroin)

        skaalattu_kuva = leikattu_kuva.resize((uus_leveys, uus_korkeus), resample=Image.Resampling.LANCZOS)

        uus_kuva = Image.new("L", (28, 28), color=0)

        x_offset = (28 - uus_leveys) // 2
        y_offset = (28 - uus_korkeus) // 2
        uus_kuva.paste(skaalattu_kuva, (x_offset, y_offset))

        pikselit = np.array(uus_kuva) / 255.0

        syote = pikselit.reshape(-1, 1)

        _, _, _, oletus = kytkos(self.w1, self.bias1, self.w2, self.bias2, syote)
        tunnistanut = np.argmax(oletus)
        todennakoisyys = np.max(oletus) * 100

        self.label_tulos.config(text=f"Verkko tunnisti {tunnistanut} varmuudella {todennakoisyys:.2f}%")


if __name__ == "__main__":
    '''
    Pääohjelma, joka toimii valmiiksi treenautetulla .npz tiedostolla
    '''
    tiedosto = 'painot3.npz'

    try:
        parametrit = np.load(tiedosto)

        w1 = parametrit['w1']
        bias1 = parametrit ['bias1']
        w2 = parametrit ['w2']
        bias2 = parametrit ['bias2']

        app = kayttoliittyma(w1, bias1, w2, bias2)
    
    except FileNotFoundError:
        print("Tiedostoa ei löytynyt")
    
    except KeyError as ke:
        print("Tiedostosta puuttuu parametri " + ke)


