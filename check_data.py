"""Utility script to check dataset integrity."""

import hashlib

from utils.paths import DATA

BUF_SIZE = 2**20  # Hash in chunks of 1MB

DATASET_SHA256 = {
    "clevr-style.hdf5": "6bbd0b25f21ae42747f5baeb10a702c3952ed9f41b2ba5bc67cf7c4149939324",
    "clevr_10-full.hdf5": "419dd56bc7791541daaa246d4b60a00ab3d9c0b9a1330c8b8a015350130240c2",
    "clevr_10-metadata.npy": "d29017b07e9f59c746706ab679d32540eb33ce5582028d59b4594c1a6a850daf",
    "multidsprites-style.hdf5": "b1c2ee9241ec43955817ef0b5911acf6a7eb40be88d34c030e07373729ac3f4c",
    "multidsprites_colored_on_grayscale-full.hdf5": "a16e5777790adce45e83734b9693061b84725b341fb900439fe64bb08d9b2530",
    "multidsprites_colored_on_grayscale-metadata.npy": "9cbbbf1212ad9e5c80e31fd1658ccd953065513ee2a55c3ca6adab09ba74c68a",
    "objects_room-style.hdf5": "90117a3dca72f157598b5d8f7c9df938366f4a203c0c78911dcf161cf4b96ade",
    "objects_room_train-full.hdf5": "5bddda03ebb48dca426071317c3d3ed1fb1a6683fd20c5fdf57a0a8c04bf2444",
    "objects_room_train-metadata.npy": "8c63f928ff46667d8362e2fda1dee38d3e152877c2c9bdae80acfed82632c985",
    "shapestacks-full.hdf5": "09433b2e629744fc2aaa81d5f922d8b0509500bcb0e49d884675f4f8d52c1928",
    "shapestacks-metadata.npy": "bb975905693c706d91c338d66a22fa6761ca37472d0641482446018eaaa0c2be",
    "shapestacks-style.hdf5": "c5088bdf79ce0950610007813ac80c77bf35af0b4dcd3b341d27cd7fdd6e99e7",
    "tetrominoes-full.hdf5": "de19edfe00e6322058edbad25c9b3a8c55f7112cf9dac1565ccd87a2cdb87ffd",
    "tetrominoes-metadata.npy": "39ec2179213e1281e24d2f0869751846e8dd065bbcf5e1f1b435d967b73824fd",
    "tetrominoes-style.hdf5": "fcaeeb7656b6fc83d7108af33063772b605d5a22c58e2fb27eb7c28dec4ec22e",
    "clevrtex-full.hdf5": "7e1a667a2587bc64509eeab96082c15c22d53105be2aa74495277db33f3f839f",
    "clevrtex-metadata.npy": "ec28e2ef7a19469ee3b0c187af42ca472c15074f1cea12db6f1378fa3fcbf948",
}

ignored = [".DS_Store", "LICENSE", "README.txt"]

if __name__ == "__main__":
    print(f"Checking files in '{DATA}'...")
    total = 0
    unmatched = []
    for filename in DATA.iterdir():
        name = filename.name
        if name in ignored:
            continue
        total += 1
        if name not in DATASET_SHA256:
            print(f"  {name:60s}  SKIPPED (unrecognized file name)")
            unmatched.append(name)
            continue

        sha256 = hashlib.sha256()
        with open(filename, "rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        digest = sha256.hexdigest()

        if digest == DATASET_SHA256[name]:
            msg = "matched"
        else:
            unmatched.append(name)
            msg = "NOT MATCHED"
            print(digest)
        print(f"  {name:60s}  {msg}")
    if len(unmatched) == 0:
        print("All files were matched")
    else:
        print(f"{len(unmatched)} files (out of {total}) were not matched:")
        for name in unmatched:
            print(f"  {name}")
