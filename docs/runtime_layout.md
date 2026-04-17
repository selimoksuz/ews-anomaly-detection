## Runtime Layout

Bu projede calisma zamaninda uretilen dosyalar tek bir kok altinda toplanir:

- `runtime/registry`
  - `run_registry.json`: tum run kayitlari
  - `model_registry.json`: candidate/champion model kayitlari
  - `champions.json`: segment bazli aktif champion pointer'lari
- `runtime/runs/<run_id>`
  - `manifest.json`: ilgili run'in ana kaydi
  - `logs/`: run'a ait development, scoring, monitoring veya operations loglari
  - `monitoring/`: run'a ait monitoring bundle dosyalari
- `runtime/models`
  - gercekten artifact ureten run'larin tekrar kullanilabilir dosyalari
  - ornek: `model.pkl`, `calibration.json`, `stability.json`, `feature_selection.json`, `weights.json`, `evaluation.json`
- `runtime/logs`
  - yalnizca CLI oturum loglari (`cli/`)

### Tasarim Kurallari

- CLI ve notebook ayni runtime kokunu kullanir.
- Runtime path'leri `cwd`'ye gore degil, repo kokune gore resolve edilir.
- Notebook calistirmak `notebooks/meta`, `notebooks/artifacts` veya benzeri paralel klasorler uretmemelidir.
- Run bazli degerlendirme icin asil referans klasor `runtime/runs/<run_id>` altidir.
- Tek bir run'a ait manifest, log ve monitoring dosyalari ayni klasorde toplanir.

### Legacy Ayrimi

Asagidaki yapilar aktif lifecycle akisinin parcasi degildir:

- `models/`
- `output/`

Bu klasorler eski `EWSPipeline` denemelerinden kalmistir. Guncel lifecycle ve Oracle bazli akista kullanilan resmi yerlesim `runtime/` altidir.
