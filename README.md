# cloud-net-rfdetr

Types de nuages + segmentation sémantique + instances, en quasi temps réel, sur **Meteosat Europe** (et GOES-19).

Le live utilise un **U-Net** entraîné sur les produits officiels **GOES-19** (truecolor CMI + phase ACTP + épaisseur COD + hauteur ACHA), puis fine-tuné sur **Meteosat Europe** ([MET Norway](https://api.met.no/weatherapi/geosatellite/1.4/documentation)).

Les instances sont obtenues par composantes connexes **par type** (pas un détecteur d’objets séparé).

<p align="center">
  <img src="docs/screenshots/dashboard-europe-instances.png" alt="Dashboard live Europe" width="100%">
</p>

## Types

| Classe | Signification |
|---|---|
| Brouillard / très bas | sommet < 1 km |
| Nuages bas | St / Sc / Cu |
| Nuages moyens | Ac / As |
| Phase mixte | eau + glace |
| Hauts opaques | Cs / Ns |
| Très hauts / convectif | Cb |
| Cirrus très mince / cirrus / cirrus épais | glace, selon l’épaisseur optique |
| Clair / inconnu | ciel clair ou retrieval impossible |

Ce n’est **pas** la taxonomie WMO complète (un cumulus vs un stratus). GOES/Meteosat donnent l’étage + la phase, pas la forme du nuage.

<p align="center">
  <img src="docs/screenshots/europe-brut.png" alt="Meteosat Europe brut" width="32%">
  <img src="docs/screenshots/europe-semantic.png" alt="Masque semantique" width="32%">
  <img src="docs/screenshots/europe-instances.png" alt="Instances" width="32%">
</p>

## Pourquoi pas du 30 m toutes les 10 min

**95-Cloud** = Landsat 8, **30 m**, patches **384×384** (~11,5 km). Revisit : plusieurs jours.

Le live géostationnaire (Meteosat / GOES) = **0,5–2 km**, **2,5–15 min**. Aucun satellite public ne combine les deux. Le dashboard sert à la cinétique ; Landsat/Sentinel restent la haute résolution.

Un premier essai RF-DETR-Seg sur 38/95-Cloud (binaire, instances Landsat) ne se transfère pas au live : écart de domaine trop fort.

## Installation

```powershell
cd C:\Users\Remy\cloud-net-rfdetr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch CUDA : installer la wheel CUDA correspondant à ta carte, par exemple :

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Dashboard

```powershell
streamlit run app/streamlit_app.py
```

Ouvre [http://127.0.0.1:8501](http://127.0.0.1:8501). Source par défaut : **Meteosat Europe** (visible le jour, infrared la nuit).

Les poids locaux `runs/unet-cloud-types-europe/best.pt` sont chargés s’ils existent. Ils ne sont **pas** dans Git (trop lourds).

### Sources live

| id | flux | cadence |
|---|---|---|
| `europe` | MET Norway Meteosat Europe, visible / IR auto | ~15 min |
| `europe-visible` | MET Norway visible | ~15 min |
| `europe-ir` | MET Norway infrared | ~15 min |
| `meteosat-europe` | CIRA SLIDER GeoColor, crop Europe | variable |
| `meteosat-disk` | CIRA SLIDER GeoColor disque | ~15 min |
| `goes-19-cmi` | NOAA GOES-19 ABI CMI CONUS (même domaine que l’entraînement) | ~5–10 min |
| `goes-19` | CIRA SLIDER GOES-19 GeoColor | ~10 min |
| `goes-19-star` | NOAA STAR GeoColor full disk | ~10 min |

## Entraînement (U-Net types)

Télécharge des paires GOES-19 publics (AWS NOAA, sans clé) : MCMIP + ACTP + COD + ACHA.

```powershell
python scripts/download_goes_types.py --max-scenes 24 --days 10
python -c "from src.ingest.goes_aws import download_acha_for_existing; print(len(download_acha_for_existing()))"
python scripts/prepare_goes_seg.py
python scripts/train_cloud_types.py --epochs 20 --batch-size 8
python scripts/finetune_europe.py --max-images 36 --epochs 10
```

Checkpoints :

- `runs/unet-cloud-types/best.pt` — GOES CONUS
- `runs/unet-cloud-types-europe/best.pt` — fine-tune Meteosat Europe

## Pipeline historique (38-Cloud / 95-Cloud + RF-DETR)

Toujours dans le dépôt, plus utilisé par le dashboard.

```powershell
python scripts/download_datasets.py
python scripts/prepare_coco.py --max-train 0
python scripts/train_rfdetr.py --epochs 20
```

- [38-Cloud](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) / [95-Cloud](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset)
- Kaggle : jeton dans `%USERPROFILE%\.kaggle\kaggle.json`
- Sans jeton : miroirs Hugging Face (`jaygala223/38-cloud-dataset`, `jaygala223/95-cloud-train-only-v1`)

## Structure

```
app/streamlit_app.py          dashboard
src/unet.py                   U-Net
src/cloud_types.py            taxonomie 11 classes
src/infer_types.py            inference
src/instances.py              instances (composantes connexes)
src/ingest/                   MET Norway, SLIDER, GOES AWS, STAR
scripts/                      download / prepare / train / finetune
docs/screenshots/             captures du dashboard
data/                         ignore (datasets locaux)
runs/                         ignore (poids)
```

## Licence des données

- Images live MET Norway : conditions [api.met.no](https://api.met.no/)
- GOES-19 L2 : NOAA, domaine public
- 38/95-Cloud : voir les dépôts SorourMo
- CIRA SLIDER : usage selon CIRA / RAMMB
