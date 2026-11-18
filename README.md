# EDE Sustainable VRPTW

Tek aktif araştırma projesi. Çalışma, uygulanabilir çözümler arasında E-DE'nin model tabanlı bileşik amaç değerini dört dondurulmuş iç karşılaştırıcıyla değerlendirir. Ana kanıt 36 örnek × 3 senaryo × 5 yöntem × 20 eşleştirilmiş tohum = 10.800 kayıttır. Doğrulayıcı bileşen ablasyonu 48/1.080 kayıtta açıktır.

GitHub'daki tek dal `main`'dir. Depo; kod, testler, belgeler, kilitli bağımlılıklar ve türetilmiş özetleri içerir. Ham deney sonuçları, birincil koşu manifesti, benchmark kopyaları, makale, gönderim paketleri ve tarihsel arşiv tam yerel projede tutulur. Aşağıdaki yapı bu tam çalışma alanını açıklar; yerel araştırma klasörleri GitHub klonunda bulunmaz.

| Klasör | Sorumluluk |
|---|---|
| `src/swevo_suite/` | Model, çözücü ve karşılaştırıcı uygulamaları |
| `scripts/` | Doğrulama, analiz ve yeniden üretim komutları |
| `tests/` | Birim ve bilimsel kayıt regresyon testleri; sentetik girdiler |
| `configs/` | Kilitli deney tasarımı ve yöntem ayarları |
| `data/primary/` | Değiştirilmemiş ana sonuç defteri |
| `data/secondary/` | Yerel kilitli ikincil kanıt |
| `data/reference/` | Yayımlanan dondurulmuş karşılaştırma referansı |
| `data/ablation/` | Özgün ön kayıt, manifest, kaynak kilidi ve kısmi ham sonuç |
| `data/benchmarks/` | Yerel kamu benchmark önbelleği; gönderim ekine dahil edilmez |
| `results/primary/`, `results/ablation/` | Yeniden üretilen tablolar, çıkarımlar, durum ve köken kaydı |
| `manuscript/` | Makalenin aktif LaTeX kaynağı, PDF ve eşlik eden belgeler |
| `submission/` | Tek güncel gönderim paketi ve doğrulama raporu |
| `docs/` | Yöntem, yeniden üretim, A–Z denetimi ve açık işler |
| `archive/` | Geçmiş dondurulmuş kanıtlar ve bakım kayıtları; aktif çalışma değildir |

## Çalıştırma

Proje kökünde Python 3.11 veya üstü kullanın. Doğrulanan ortam: CPython 3.14.3, Windows 11 build 26200. Bağımlılıklar `requirements.lock.txt` içindedir.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock.txt
.venv/Scripts/python -m pytest -q -p no:cacheprovider
```

Tam yerel proje girdileriyle aşağıdaki komutlar kullanılabilir. GitHub klonu tek başına ham girdileri içermez. Hakem eki, `--core` için gereken girdileri bağımsız olarak sağlar; ablasyonun yeniden denetimi ve portal kontrolü tam yerel projeye aittir.

```powershell
.venv/Scripts/python scripts/finalize.py --core
.venv/Scripts/python scripts/audit_ablation.py --campaign data/ablation --out results/ablation
.venv/Scripts/python scripts/finalize.py --portal
```

`--core` kayıtlı sonuçlardan 41 analiz çıktısını üretir; yeni optimizasyon koşusu başlatmaz. Hakem eki `--core` ve testler için gereken girdileri içerir; makale ve tarihsel arşiv gerekmez. Portal kontrolü ayrı çalışır. Ayrıntılar: [yeniden üretim](docs/REPRODUCIBILITY.md), [A–Z denetimi](docs/AUDIT_REPORT.md), [ablasyona devam](docs/ABLATION_CONTINUATION.md).

Ana sonuç: her yöntem tüm kayıtlarda uygulanabilir ve sıfır fazla mesailidir. E-DE'nin en iyi diğer dondurulmuş karşılaştırıcıya göre küçük/orta/büyük katman bileşik skor düşüşleri %4,30 / %14,64 / %15,48'dir. Bunlar saha tasarrufu veya tekil bileşen etkisi değildir. 108 örnek×senaryo hücresi aynı 36 örneği tekrarlar. Nominal arama sınırı ile +10 saniyelik kayıt toleransı ayrıdır.

Tam yerel projede güncel makale `submission/manuscript.pdf` dosyasıdır. Gönderilecek dosyalar ve SHA-256 toplamları yalnız `submission/` altındadır. Tarihsel dosyalar baytları korunarak `archive/history/` altına taşınmıştır. Dergiye gönderim yapılmamıştır.
