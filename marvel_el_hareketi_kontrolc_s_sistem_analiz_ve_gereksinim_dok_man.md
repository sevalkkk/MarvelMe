# PROJE ANALİZ VE SİSTEM GEREKSİNİM DOKÜMANI (SRS)
**Proje Adı:** Marvel Gesture Switcher (Sıfır Hata Paylı El Hareketi ile Karakter Değiştirici)  
**Hedef Geliştirme Ortamı:** Antigravity IDE / Python Geliştirme Ortamı  
**Mimari Yaklaşım:** Gerçek Zamanlı Bilgisayarlı Görü (Real-Time Computer Vision) ve Durum Makinesi (State Machine)

---

## 1. PROJENİN AMACI VE KAPSAMI
Bu projenin amacı, bilgisayar kamerasından alınan gerçek zamanlı görüntü üzerinden kullanıcının el iskeletini (hand landmarks) takip etmek ve **başparmak ucu ile orta parmak ucunun birbirine temas etmesi (vurulması/şıklatılması)** hareketini "sıfıra yakın hata payıyla" algılayarak ekranda gösterilen Marvel karakter görsellerini sırasıyla değiştirmektir.

Yüklenen arşivdeki (`WhatsApp Unknown 2026-11-18 at 20.06.23.zip`) 3 adet görsel, başlangıç veri seti olarak kullanılacak ve her parmak vuruşunda bu görseller arasında pürüzsüz bir geçiş yapılacaktır.

---

## 2. "SIFIR HATA PAYI" İÇİN ALGORİTMİK GEREKSİNİMLER (ÇOK KRİTİK)
Standart piksel mesafesi ölçümleri, el kameraya yaklaştığında veya uzaklaştığında hatalı sonuçlar üretir. Hata payını ortadan kaldırmak için aşağıdaki **4 Katmanlı Doğrulama Algoritması** uygulanmalıdır:

### 2.1. Dinamik Ölçekleme (Scale-Invariant Normalization)
El kameraya yaklaştığında parmaklar arası piksel mesafesi büyür, uzaklaştığında küçülür. Sabit piksel eşiği (ör. `35px`) kullanmak hataya yol açar.
* **Çözüm:** Başparmak ucu (`Landmark 4`) ile Orta parmak ucu (`Landmark 12`) arasındaki Öklid mesafesi ($D_{tetik}$), elin o anki büyüklüğüne oranlanmalıdır.
* **Referans Ölçek ($D_{ref}$):** Bilek (`Landmark 0`) ile Orta parmak kökü/eklemi (`Landmark 9`) arasındaki mesafe elin sabit kemik uzunluğunu verir.
* **Formül:** 
  $$Oran = \frac{D_{tetik}}{D_{ref}}$$
* Bu oran kameraya olan uzaklıktan bağımsızdır. $Oran < 0.22$ olduğunda temas gerçekleşmiş kabul edilir.

### 2.2. Çift Eşikli Histerezis (Schmitt Trigger / Hysteresis)
Parmaklar tam sınır noktasında titrediğinde sistemin art arda 5-10 kez karakter değiştirmesini önlemek için tek eşik yerine iki ayrı eşik kullanılmalıdır:
* **Tetikleme Eşiği (`TRIGGER_THRESHOLD = 0.22`):** Oran bu değerin **altına** düştüğünde karakter değişir ve sistem `KİLİTLİ (LOCKED)` durumuna geçer.
* **Serbest Bırakma Eşiği (`RELEASE_THRESHOLD = 0.40`):** Yeni bir vuruşun algılanabilmesi için parmakların birbirinden iyice ayrılması, yani oranın `0.40`'ın **üzerine** çıkması şarttır.

### 2.3. Hareket Yumuşatma (Exponential Moving Average - EMA)
Kameradaki anlık ışık dalgalanmaları veya sensör gürültüsü (noise) nedeniyle eklem noktalarının milimetrik sıçramalar yapmasını engeller.
* **Formül:** 
  $$S_t = \alpha \cdot Y_t + (1 - \alpha) \cdot S_{t-1}$$
* Burada $Y_t$ anlık ölçülen oran, $S_t$ ise yumuşatılmış orandır ($\alpha = 0.5$ seçilmelidir).

### 2.4. Zaman Koruması (Cooldown / Debounce Timer)
İki başarılı karakter değişimi arasında minimum **400 milisaniye (0.4 saniye)** zorunlu bekleme süresi bulunmalıdır.

---

## 3. FONKSİYONEL GEREKSİNİMLER
1. **Varlık Yönetimi (Asset Loader):**
   * Uygulama çalışırken `assets/` klasörü içerisindeki tüm `.jpg`, `.jpeg` ve `.png` dosyalarını (zip'ten çıkan 3 adet görsel dahil) otomatik taramalı ve hafızaya (RAM) önceden yüklemelidir. Karakter geçişlerinde diskten okuma yapılarak gecikme (lag) yaşanmamalıdır.
2. **Arayüz (UI/HUD) ve Görüntüleme:**
   * Ekran tam boyutta (örneğin `1280x720`) çalışmalı ve ana ekranda aktif Marvel karakterinin görseli yüksek kalitede gösterilmelidir.
   * Ekranın sağ veya sol alt köşesinde **Picture-in-Picture (PiP)** formatında küçük bir pencerede (`320x180`) kullanıcının kamera görüntüsü, el iskeleti ve parmaklar arasındaki mesafe çizgisi canlı olarak gösterilmelidir.
   * Parmaklar birbirine yaklaştığında çizgi rengi **Yeşilden -> Kırmızıya** dönerek kullanıcıya görsel geri bildirim vermelidir.
3. **Klavye Kısayolları:**
   * `ESC` veya `Q`: Uygulamadan güvenli çıkış.
   * `R`: Karakter listesini başa sarma (Reset).

---

## 4. TEKNOLOJİ YIĞINI VE ORTAM KURULUMU
* **Dil:** Python 3.9 - 3.11 (MediaPipe uyumluluğu için en stabil sürümler)
* **Kütüphaneler:**
  * `opencv-python`: Görüntü işleme, arayüz çizimi ve kamera kontrolü.
  * `mediapipe`: Yüksek doğruluklu el ve parmak eklemi tespiti.
  * `numpy`: Hızlı matris ve matematiksel mesafe hesaplamaları.

---

## 5. PROJE KLASÖR YAPISI
Antigravity IDE üzerinde oluşturulacak proje dizini şu şekilde olmalıdır:

```text
marvel_gesture_project/
│
├── assets/                                      # Marvel görsellerinin bulunacağı klasör
│   ├── WhatsApp Image 2026-11-18 at 19.58.36.jpeg
│   ├── WhatsApp Image 2026-11-18 at 19.58.36 (1).jpeg
│   └── WhatsApp Image 2026-11-18 at 19.58.37.jpeg
│
├── src/
│   ├── __init__.py
│   ├── gesture_detector.py                      # MediaPipe, normalizasyon ve histerezis motoru
│   ├── character_manager.py                     # Görselleri RAM'de tutan ve geçiş yapan sınıf
│   └── ui_renderer.py                           # Ana ekran ve PiP kamera görüntüsünü birleştiren sınıf
│
├── main.py                                      # Uygulamayı başlatan ana dosya
└── requirements.txt                             # Gerekli kütüphaneler listesi
```

---

## 6. ANTIGRAVITY IDE İÇİN DOĞRUDAN KULLANILABİLİR "MASTER PROMPT"
Projeyi Antigravity IDE'de tek seferde ve hatasız kodlatmak için aşağıdaki metni kopyalayıp IDE'nin AI asistanına yapıştırabilirsin:

> **ANTIGRAVITY IDE PROMPTU:**
> "Python, OpenCV ve MediaPipe kullanarak gerçek zamanlı çalışan bir 'Marvel Karakter Değiştirici' uygulaması yazmanı istiyorum. Proje şu teknik kurallara kesinlikle uymalıdır:
> 1. `assets/` klasöründeki görselleri (`jpeg`, `png`) başlangıçta RAM'e yükleyen ve `1280x720` çözünürlüğe ölçekleyen bir `CharacterManager` sınıfı oluştur.
> 2. MediaPipe Hands (`max_num_hands=1`, `min_detection_confidence=0.8`, `min_tracking_confidence=0.8`) kullanan bir `GestureDetector` sınıfı yaz.
> 3. Başparmak ucu (Landmark 4) ile Orta parmak ucu (Landmark 12) arasındaki Öklid mesafesini hesapla. Kameraya uzaklıktan etkilenmemesi için bu mesafeyi, Bilek (Landmark 0) ile Orta parmak kökü (Landmark 9) arasındaki mesafeye bölerek normalize et.
> 4. Hesaplanan normalize orana $\alpha = 0.5$ ile Exponential Moving Average (EMA) filtresi uygula.
> 5. Hatalı ve çoklu tetiklemeleri tamamen sıfırlamak için Schmitt Trigger (Histerezis) durum makinesi kullan: Oran `0.22`'nin altına düştüğünde karakteri değiştir ve kilitle (`is_locked = True`). Oran tekrar `0.40`'ın üzerine çıkmadan kilidi açma (`is_locked = False`). Ayrıca iki tetikleme arasına `0.4` saniye cooldown ekle.
> 6. Ana pencerede (`1280x720`) aktif Marvel karakterini tam ekran göster. Sağ alt köşeye `320x180` boyutunda Picture-in-Picture (PiP) olarak canlı kamera görüntüsünü, el iskeletini ve Landmark 4 ile 12 arasındaki çizgiyi yerleştir.
> 7. Tüm kodu modüler, temiz, PEP8 standartlarına uygun ve doğrudan çalıştırılabilir şekilde oluştur."