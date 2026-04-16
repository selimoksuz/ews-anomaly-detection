## Runtime Layout

Bu projede calisma zamaninda uretilen dosyalar tek bir kok altinda toplanir:

- `runtime/registry`
  - `run_registry.json`: tum run kayitlari
  - `model_registry.json`: candidate/champion model kayitlari
  - `champions.json`: segment bazli aktif champion pointer'lari
  - `runs/<run_id>/manifest.json`: run manifest ve run-level monitoring dosyalari
- `runtime/models`
  - segment ve run bazli model artifact'leri
  - calibration, evaluation, stability, comparison ve benzeri dosyalar
- `runtime/logs`
  - CLI ve pipeline log dosyalari
- `runtime/monitoring`
  - toplu monitoring export'lari icin ayrilmis alan

### Tasarim Kurallari

- CLI ve notebook ayni runtime kokunu kullanir.
- Runtime path'leri `cwd`'ye gore degil, repo kokune gore resolve edilir.
- Notebook calistirmak `notebooks/meta`, `notebooks/artifacts` veya benzeri paralel klasorler uretmemelidir.

### Legacy Ayrimi

Asagidaki yapilar aktif lifecycle akisinin parcasi degildir:

- `models/`
- `output/`

Bu klasorler eski `EWSPipeline` denemelerinden kalmistir. Guncel lifecycle ve Oracle bazli akista kullanilan resmi yerlesim `runtime/` altidir.
