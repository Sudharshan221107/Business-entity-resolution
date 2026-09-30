import zipfile
import os
import sys

def create_submission_zip(team_name="team_resolution"):
    if len(sys.argv) > 1 and sys.argv[1].strip():
        team_name = sys.argv[1].strip()

    clean_team_name = team_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if clean_team_name.endswith("_submission.zip"):
        zip_filename = clean_team_name
    elif clean_team_name.endswith(".zip"):
        zip_filename = clean_team_name.replace(".zip", "_submission.zip")
    else:
        zip_filename = f"{clean_team_name}_submission.zip"

    files_to_pack = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("code/business_entity_resolution/run.py", "code/business_entity_resolution/run.py"),
        ("code/business_entity_resolution/tests/test_pipeline.py", "code/business_entity_resolution/tests/test_pipeline.py"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    src_dir = os.path.join("code", "business_entity_resolution", "src")
    if os.path.exists(src_dir):
        for f in os.listdir(src_dir):
            if f.endswith(".py"):
                disk_path = os.path.join(src_dir, f)
                arc_path = f"code/business_entity_resolution/src/{f}"
                files_to_pack.append((disk_path, arc_path))

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for disk_path, arc_path in files_to_pack:
            if os.path.exists(disk_path):
                zf.write(disk_path, arc_path)
            else:
                print(f"WARNING: File not found: {disk_path}")

    print(f"Successfully generated submission ZIP: {zip_filename}")
    print("Archive verification:")
    with zipfile.ZipFile(zip_filename, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size} bytes)")

if __name__ == "__main__":
    create_submission_zip()
