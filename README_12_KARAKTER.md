# MarvelMe — 12 Karakterli Çizgi Roman Sürümü

Bu sürüm, önceki dört karaktere Thor, Loki, Black Panther, Deadpool, Captain America, Wolverine, Venom ve Ghost Rider ekler. Portalın içindeki kamera görüntüsüne çizgi roman gölgelendirmesi uygulanır; yüz algılandığında karaktere özgü çizilmiş maske/kask ve aksesuarlar yüz landmark'larını takip eder.

**Önemli sınır:** Bu sürümde film veya çizgi roman karakterinin eksiksiz 3D yüzü, saçları, tüm vücudu ve kostümüyle birebir değişimi yoktur. Arka plan da tamamen yeni bir Marvel mekânına dönüşmez; mevcut odanın çizgi roman estetiğiyle işlenmiş hâli görünür. Tam vücut dönüşümü ve sahne üretimi ayrı bir model/varlık geliştirme aşamasıdır.

## Windows kurulumu

1. ZIP'i açıp `MarvelMe` klasöründe VS Code terminali açın.
2. `py -3.12 -m venv .venv`
3. `.venv\Scripts\python.exe -m pip install -r requirements.txt`
4. `.venv\Scripts\python.exe main.py --face-filter`

Kamera ve MediaPipe gereklidir. İki el görünürken portal oluşur. El hareketiyle karakter değişir. `F` tuşu yüz filtresini açıp kapatır. Başka tuşlar ve kullanım bilgileri için `TECHNICAL_DOCUMENTATION.md` dosyasına bakın.

## Test durumu

Python sözdizimi ve mevcut dört regresyon testi bu çalışma ortamında geçti. Kamera, Windows sürücüleri, tüm 12 maskenin gerçek yüzle uyumu ve gerçek FPS bu ortamda doğrulanamadı. Bu nedenle sıfır hata veya belirli FPS garantisi verilmez.
