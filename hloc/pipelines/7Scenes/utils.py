import logging

import pycolmap

logger = logging.getLogger(__name__)


def create_reference_sfm(full_model, ref_model, blacklist=None):
    """Create a new COLMAP model with only training images."""
    logger.info("Creating the reference model.")
    ref_model.mkdir(exist_ok=True, parents=True)
    full = pycolmap.Reconstruction(full_model)

    blacklist_set = set()
    if blacklist is not None:
        with open(blacklist, "r") as f:
            blacklist_set = set(f.read().rstrip().split("\n"))

    ref = pycolmap.Reconstruction()
    for camera in full.cameras.values():
        ref.add_camera(camera)
    for rig in full.rigs.values():
        ref.add_rig(rig)

    kept_image_ids = set()
    kept_frame_ids = set()
    for id_, image in full.images.items():
        if image.name in blacklist_set:
            continue
        kept_image_ids.add(id_)
        if image.has_frame_id():
            kept_frame_ids.add(image.frame_id)

    for frame_id in kept_frame_ids:
        frame = full.frames[frame_id]
        new_frame = pycolmap.Frame(
            frame_id=frame.frame_id,
            rig_id=frame.rig_id,
        )
        for data_id in frame.data_ids:
            if (
                data_id.sensor_id.type != pycolmap.SensorType.CAMERA
                or data_id.id in kept_image_ids
            ):
                new_frame.add_data_id(data_id)
        if frame.has_pose():
            new_frame.rig_from_world = frame.rig_from_world
        ref.add_frame(new_frame)

    for id_ in kept_image_ids:
        image = full.images[id_]
        new_image = pycolmap.Image(
            image_id=image.image_id,
            camera_id=image.camera_id,
            frame_id=image.frame_id,
            name=image.name,
            points2D=pycolmap.Point2DList(image.points2D),
        )
        for idx in new_image.get_observation_point2D_idxs():
            new_image.reset_point3D_for_point2D(idx)
        ref.add_image(new_image)

    for id_, point3D in full.points3D.items():
        ref_elements = [
            el for el in point3D.track.elements if el.image_id in kept_image_ids
        ]
        if len(ref_elements) == 0:
            continue
        ref.add_point3D_with_id(
            id_,
            pycolmap.Point3D(
                xyz=point3D.xyz,
                color=point3D.color,
                error=point3D.error,
                track=pycolmap.Track(ref_elements),
            ),
        )

    ref.write_binary(ref_model)
    logger.info(f"Kept {len(kept_image_ids)} images out of {len(full.images)}.")
