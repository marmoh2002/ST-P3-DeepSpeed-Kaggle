import os
import json
import argparse
from collections import defaultdict

def extract_logfile_from_filename(filename):
    """
    Extracts the logfile part from a typical NuScenes filename.
    Example: 'n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151061547405.jpg'
    Returns: 'n008-2018-08-01-15-16-36-0400' or None if format doesn't match.
    """
    if '__' in filename:
        return filename.split('__')[0]
    return None

def find_scenes_for_logfiles(present_logfiles, log_table, scene_table):
    """
    Finds scene names corresponding to a set of logfile names.
    """
    print(f"\nMapping {len(present_logfiles)} present logfiles to log tokens...")
    logfile_to_token = {}
    found_tokens = set()
    log_token_counts = defaultdict(int) # For debugging/interest

    for log_rec in log_table:
        logfile = log_rec.get('logfile')
        log_token = log_rec.get('token')
        if logfile and log_token and logfile in present_logfiles:
            logfile_to_token[logfile] = log_token
            found_tokens.add(log_token)

    print(f"Found, {len(logfile_to_token)} matching log records in log.json.")
    missing_logs = present_logfiles - set(logfile_to_token.keys())
    if missing_logs:
        print(f"WARNING: Could not find log records for: {missing_logs}")

    print(f"\nSearching {len(scene_table)} scenes for matching log tokens...")
    relevant_scene_names = set()
    for scene_rec in scene_table:
        scene_log_token = scene_rec.get('log_token')
        scene_name = scene_rec.get('name')
        if scene_log_token and scene_name and scene_log_token in found_tokens:
            relevant_scene_names.add(scene_name)
            log_token_counts[scene_log_token] += 1

    print("-" * 20)
    print("Log token usage counts (scenes per token):")
    for token, count in log_token_counts.items():
        # Find logfile name for this token for better readability
        token_logfile = "[Unknown Logfile]"
        for lf, tk in logfile_to_token.items():
            if tk == token:
                token_logfile = lf
                break
        print(f"  Log Token {token} ({token_logfile}): {count} scenes")
    print("-" * 20)


    return relevant_scene_names

def main(data_folder, metadata_folder, output_file1='target_scenes.txt',output_file2='target_logfile.txt'):
    """
    Main function to find scene names from files in a data folder.
    """
    print(f"Scanning data folder: {data_folder}")
    present_logfiles = set()
    try:
        for item_name in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item_name)
            if os.path.isfile(item_path):
                logfile = extract_logfile_from_filename(item_name)
                if logfile:
                    present_logfiles.add(logfile)
    except FileNotFoundError:
        print(f"Error: Data folder not found: {data_folder}")
        return
    except Exception as e:
        print(f"Error scanning data folder: {e}")
        return

    if not present_logfiles:
        print("No files matching the expected NuScenes format (logfile__sensor__...) found.")
        return

    print(f"\nFound {len(present_logfiles)} unique potential logfile names from filenames:")
    for logfile in sorted(list(present_logfiles)):
        print(f"  - {logfile}")

    log_json_path = os.path.join(metadata_folder, 'log.json')
    scene_json_path = os.path.join(metadata_folder, 'scene.json')

    log_table = None
    scene_table = None

    # Load metadata (can cause MemoryError)
    try:
        print(f"\nLoading log metadata: {log_json_path}")
        if not os.path.exists(log_json_path):
            print(f"Error: log.json not found at {log_json_path}")
            return
        with open(log_json_path, 'r') as f:
            log_table = json.load(f)
        print(f"Loaded {len(log_table)} log records.")

        print(f"\nLoading scene metadata: {scene_json_path}")
        if not os.path.exists(scene_json_path):
            print(f"Error: scene.json not found at {scene_json_path}")
            return
        with open(scene_json_path, 'r') as f:
            scene_table = json.load(f)
        print(f"Loaded {len(scene_table)} scene records.")

    except MemoryError:
        print("\n" + "="*60)
        print("Error: Ran out of memory trying to load log.json or scene.json.")
        print("This script requires loading these full metadata files.")
        print("You may need to run this lookup on a machine with more RAM,")
        print("or perform the logfile->scene lookup offline (see previous suggestions).")
        print("="*60)
        return
    except FileNotFoundError as e:
        print(f"Error: Metadata file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Find the corresponding scenes
    relevant_scenes = find_scenes_for_logfiles(present_logfiles, log_table, scene_table)

    print("\n" + "="*60)
    if not relevant_scenes:
        print("No scenes found corresponding to the logfiles extracted from the data folder.")
    else:
        print("Scene names corresponding to files found in the data folder:")
        for scene_name in sorted(list(relevant_scenes)):
            print(f"  - {scene_name}")
        print(f"\nTotal unique scenes found: {len(relevant_scenes)}")

        # Write scene names to a file
        try:
            with open(output_file1, 'w') as f:
                print("scene names:")
                for scene_name in sorted(list(relevant_scenes)):
                    f.write(scene_name + '\n')
            print(f"\nScene names written to: {output_file1}")
        except Exception as e:
            print(f"Error writing to {output_file1}: {e}")
        try:
            with open(output_file2, 'w') as f:
                print("logfile names:")    
                for logfile in sorted(list(present_logfiles)):
                    f.write(logfile + '\n')
            print(f"\nlog names written to: {output_file2}")
        except Exception as e:
            print(f"Error writing to {output_file2}: {e}")
    print("="*60)
    print("\nUse this list of scene names for your offline JSON filtering process.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find NuScenes scene names based on data files present in a folder.")
    parser.add_argument("data_folder",
                        help="Path to the folder containing the NuScenes data files (e.g., samples/CAM_FRONT).")
    parser.add_argument("metadata_folder",
                        help="Path to the folder containing the *full* original metadata JSON files (log.json, scene.json) for the corresponding NuScenes version (e.g., v1.0-trainval).")
    parser.add_argument("--output_file1", default='target_scenes.txt',
                        help="Output file to write the scene names to (default: target_scenes.txt).")
    parser.add_argument("--output_file2", default='target_logfile.txt',
                        help="Output file to write the log names to (default: target_logfile.txt).")

    args = parser.parse_args()

    main(args.data_folder, args.metadata_folder, args.output_file1, args.output_file2)