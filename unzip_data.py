from pathlib import Path
import zipfile

if __name__ == "__main__" : 

    zip_path = Path("./mmrec_datasets.zip")
    out_dir = Path("./datasets")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)  # unzip everything

    zip_path.unlink()
    
    print(f"Unzipped to: {out_dir.resolve()}")
    
    for data_name in ["sports_outdoors", "toys_games", "beauty_care", "arts"] : 
    
        zip_path = Path(f"./datasets/{data_name}_images.zip")
        out_dir = Path(f"./datasets")
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)  # unzip everything

        zip_path.unlink()
        
        print(f"Unzipped to: {out_dir.resolve()}")
