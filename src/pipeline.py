import sys
from pathlib import Path
import pycolmap

# HLoc yolunu ekle
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
hloc_path = project_root / 'Hierarchical-Localization'
if str(hloc_path) not in sys.path:
    sys.path.append(str(hloc_path))

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    pairs_from_covisibility,
    pairs_from_exhaustive,
    reconstruction,
    visualization,
)


class FeatureExtractor:
    def __init__(self, dataset_path, outputs_path):
        self.dataset_path = Path(dataset_path)
        self.outputs_path = Path(outputs_path)
        self.outputs_path.mkdir(parents=True, exist_ok=True)

        # DOSYA İSİMLERİ (Sabitler)
        self.global_feat_name = "global-feats-netvlad"
        # Senin elindeki dosya ismi
        self.local_feat_name = "feats-superpoint-n4096-r1024"

        # Tam Dosya Yolları
        self.global_feats_path = self.outputs_path / (self.global_feat_name + ".h5")
        self.local_feats_path = self.outputs_path / (self.local_feat_name + ".h5")

        self.mapping_list = []
        self.query_list = []
        self._create_image_lists()

    def _create_image_lists(self):
        print("--- Resim Listeleri Taranıyor ---")
        for p in (self.dataset_path / 'mapping').glob('**/*'):
            if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                self.mapping_list.append(p.relative_to(self.dataset_path).as_posix())

        for p in (self.dataset_path / 'query').glob('**/*'):
            if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                self.query_list.append(p.relative_to(self.dataset_path).as_posix())

        print(f"Mapping: {len(self.mapping_list)} | Query: {len(self.query_list)}")

    def extract_global_features(self):
        conf = extract_features.confs['netvlad']
        conf['output'] = self.global_feat_name

        if self.global_feats_path.exists():
            print(f"✅ Global özellikler bulundu (ATLANDI): {self.global_feats_path.name}")
            return self.global_feats_path

        print(f"🚀 Global Özellikler Çıkarılıyor...")
        return extract_features.main(
            conf, self.dataset_path, self.outputs_path,
            image_list=self.mapping_list + self.query_list
        )

    def extract_local_features(self):
        conf = extract_features.confs['superpoint_aachen']
        conf['model']['max_keypoints'] = 4096
        conf['preprocessing']['resize_max'] = 1024
        print(f"DEBUG: {self.local_feat_name} ")
        conf['output'] = self.local_feat_name

        if self.local_feats_path.exists():
            print(f"✅ Yerel özellikler bulundu (ATLANDI): {self.local_feats_path.name}")
            return self.local_feats_path

        print(f"🚀 Yerel Özellikler Çıkarılıyor...")
        return extract_features.main(
            conf, self.dataset_path, self.outputs_path,
            image_list=self.mapping_list + self.query_list
        )
def match_mapping_images_for_sfm(extractor):
    """
    3D harita oluşturmak için mapping görüntülerini non-sequential şekilde eşleştirir.
    Aachen v1.1 SIFT modeli varsa covisibility kullanır, yoksa exhaustive'e düşer.
    """
    outputs_path = extractor.outputs_path

    # 1. Mapping-Mapping çiftlerini üret
    sfm_pairs = outputs_path / "pairs-mapping.txt"
    aachen_sift_model = extractor.dataset_path / "3D-models" / "aachen_v_1_1"

    if not sfm_pairs.exists():
        if aachen_sift_model.exists():
            print("📄 Covisibility tabanli pair'ler uretiliyor...")
            pairs_from_covisibility.main(aachen_sift_model, sfm_pairs, num_matched=20)
        else:
            print("📄 Covisibility modeli bulunamadi, exhaustive pair'lere dusuluyor...")
            pairs_from_exhaustive.main(sfm_pairs, image_list=sorted(extractor.mapping_list))

    else:
        print(f"✅ Çift listesi zaten var: {sfm_pairs.name}")

    # 2. SuperGlue ile Eşleştir
    print("--- [SfM] SuperGlue ile Eşleştiriliyor (Mapping-Mapping) ---")
    match_conf = match_features.confs['superglue']
    match_conf['model']['weights'] = 'outdoor'

    sfm_matches = outputs_path / 'matches-mapping-superglue.h5'

    if not sfm_matches.exists():
        # DÜZELTME BURADA: feature_path -> features, matches_path -> matches
        match_features.main(
            match_conf,
            sfm_pairs,
            features=extractor.local_feats_path,
            matches=sfm_matches
        )
    else:
        print(f"✅ Eşleşme dosyası zaten var: {sfm_matches.name}")

    return sfm_pairs, sfm_matches


def match_and_visualize(extractor, global_feats, local_feats, visualize=True):
    outputs_path = extractor.outputs_path
    loc_pairs = outputs_path / 'pairs-query-netvlad.txt'

    if not loc_pairs.exists():
        print("\n--- Eşleşme Adayları Bulunuyor ---")
        pairs_from_retrieval.main(
            global_feats, loc_pairs, num_matched=5,
            query_list=extractor.query_list, db_list=extractor.mapping_list
        )
    else:
        print(f"\n✅ Çift listesi bulundu: {loc_pairs.name}")

    print("--- SuperGlue ile Eşleştiriliyor (Query-Mapping) ---")
    match_conf = match_features.confs['superglue']
    match_conf['model']['weights'] = 'outdoor'

    matches_path = outputs_path / (extractor.matches_name + ".h5")

    # DÜZELTME BURADA DA YAPILDI
    matches_path = match_features.main(
        match_conf, loc_pairs,
        features=local_feats,
        matches=matches_path
    )

    if visualize:
        print("\n🎨 Eşleşmeler Görselleştiriliyor...")
        vis_path = outputs_path / 'viz_matches'
        vis_path.mkdir(exist_ok=True)

        visualization.visualize_loc_from_log(
            outputs_path,
            query_list=extractor.query_list,
            n=5,
            top_k_db=1,
            seed=2,
        )
        print(f"👀 Görseller: {vis_path}")

    return matches_path


def run_sfm_reconstruction(extractor, sfm_pairs, sfm_matches):
    """
    COLMAP kullanarak 3D Harita (Sparse Reconstruction) oluşturur.
    """
    sfm_dir = extractor.outputs_path / 'sfm_map'

    if (sfm_dir / "cameras.bin").exists():
        print(f"✅ Harita zaten oluşturulmuş: {sfm_dir}")
        print("Yeniden oluşturmak istiyorsan 'outputs/sfm_map' klasörünü sil.")
        return sfm_dir

    sfm_dir.mkdir(exist_ok=True)

    print(f"\n--- [COLMAP] 3D Harita Oluşturuluyor... ---")
    print(f"Hedef Klasör: {sfm_dir}")

    mapper_options = {
        "num_threads": 16,
        "min_num_matches": 8,  # 🔥 çok önemli
        "ba_local_max_num_iterations": 25,
        "ba_global_max_num_iterations": 50,
    }

    model = reconstruction.main(
        sfm_dir,
        extractor.dataset_path,
        sfm_pairs,
        extractor.local_feats_path,
        sfm_matches,
        image_list=extractor.mapping_list,
        camera_mode="AUTO",
        mapper_options=mapper_options
    )

    print(f"✅ Harita başarıyla oluşturuldu!")
    return model
