# MarvelMe — güncellenmiş sürüm

## Windows'ta kurulum

Python 3.12 önerilir. Proje klasöründe terminal açın:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --self-test
python -m unittest test_regressions.py
python main.py
```

İsteğe bağlı: `python main.py --camera 0 --profile --face-filter`

## Bu sürümde

- MediaPipe video takibi etkinleştirildi (Solutions/Tasks uyumluluğu).
- Algılama sırası değiştiğinde el kimliği eşleştirmesi kararlı hale getirildi.
- Video kaydı durdurulurken kabul edilmiş karelerin sessizce silinmesi önlendi.
- Döndürülmüş portal köşeleri dışbükey dörtgen olarak sıralanıyor.
- Ek regresyon testleri eklendi.

## Test kapsamı

Çevrimdışı regresyon testleri çalıştırıldı ve geçti. Bu çalışma ortamında `mediapipe` kurulu olmadığından, dahili `--self-test` komutu MediaPipe aşamasına kadar ilerleyip orada durdu. Gerçek Windows web kamerası, yüz modeli ve donanım performansı bu ortamda doğrulanamadı. Bunlar için Windows bilgisayarınızda yukarıdaki testleri çalıştırın.

**Not:** Her cihazda sıfır hata veya sabit FPS garantisi verilemez. Kamera sürücüsü, donanım, ışık ve model sürümü sonuçları etkiler.
