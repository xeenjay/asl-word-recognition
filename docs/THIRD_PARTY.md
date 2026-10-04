# Sources and separately obtained assets

This repository contains project adapters, learning documentation, and aggregate results. It does not include the ASL Citizen videos, extracted landmark caches, model weights, the original papers, or a redistributed copy of PoET. A private GitHub repository does not replace the original sources' access requirements or terms.

## PoET

- Authors: Lascar et al., *PoET: Lightweight Pose Encoder Transformer for Online Sign Language Recognition*, VISAPP 2026.
- [Official repository](https://gitlab.lisn.upsaclay.fr/mtals/publications/poet)
- Pinned revision: `dca74c30d9d604cb3b92de8fc73ce91484644259`.
- [Official checkpoint and label files](https://nextcloud.lisn.upsaclay.fr/index.php/s/5joa4HqB3DBTWqm)
- [Paper](https://www.scitepress.org/Papers/2026/142375/142375.pdf)

The previously acquired snapshot has no standalone license file in its recorded 12-file manifest. We do not add a license on behalf of its authors or redistribute those files here. Review the upstream terms before using or redistributing them. `python scripts/fetch_vendor.py` explicitly obtains the pinned source from its original host, verifies the recorded archive and file hashes, and writes it into the Git-ignored `vendor/poet/` directory. It does not fetch model weights or execute downloaded code.

The expected checkpoint SHA256 is `5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0`. Acquire the checkpoint and both original label-mapping files together. Do not substitute a different model without a new, documented evaluation. The actual files are required for neural inference; a checksum alone is not a model.

## Data and other components

- [ASL Citizen dataset description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/) and [dataset terms](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/). Each member should obtain approved access through the original source. Dataset availability is separate from this repository.
- [MediaPipe Holistic](https://chuoling.github.io/mediapipe/solutions/holistic.html) supplies pretrained landmarks. It was not developed by this project.
- [scikit-learn](https://scikit-learn.org/) supplies LinearSVC, OneVsRestClassifier, StandardScaler, and RandomForestClassifier.
- [ASL-LEX](https://asl-lex.org/download.html) supplies optional auxiliary word attributes in the historical attribute experiment.
- NumPy, PyTorch, OpenCV, and other dependencies retain their respective licenses and attribution.

No new project-wide open-source license is selected in this private course repository. Any later public release should establish the team's intended license and retain applicable third-party notices.
