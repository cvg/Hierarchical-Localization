import pycolmap

reconstruction = pycolmap.Reconstruction("outputs/demo/sfm")
print(reconstruction.summary())

# convert .bin file to .txt file
reconstruction.write_text("outputs/demo/sfm")  # text format
reconstruction.export_PLY("outputs/demo/sfm/rec.ply")  # PLY format