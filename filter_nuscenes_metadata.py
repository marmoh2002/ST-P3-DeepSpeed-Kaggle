import os
import json
import argparse
from collections import defaultdict
import time
import sys
import gc # Garbage Collector

# --- Configuration ---
# Define the order to process tables based on dependencies.
# Tables providing tokens (e.g., scene) must come before tables using them (e.g., sample).
TABLE_PROCESSING_ORDER = [
    'log',                 # Provides log_token
    'map',                 # Uses log_token
    'scene',               # Uses log_token, Provides scene_token
    'sample',              # Uses scene_token, Provides sample_token
    'sensor',              # Provides sensor_token
    'calibrated_sensor',   # Uses sensor_token, Provides calibrated_sensor_token
    'ego_pose',            # Provides ego_pose_token
    'sample_data',         # Uses sample_token, ego_pose_token, calibrated_sensor_token; Provides sample_data_token
    'category',            # Provides category_token
    'attribute',           # Provides attribute_token
    'visibility',          # Provides visibility_token
    'instance',            # Uses category_token, Provides instance_token
    'sample_annotation'    # Uses sample_token, instance_token, visibility_token, attribute_tokens
]

# --- Helper Functions ---

def read_target_scenes(filepath):
    """Reads scene names from a file, one per line."""
    print(f"Reading target scene names from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            # Use a set for efficient lookup
            scenes = {line.strip() for line in f if line.strip()}
        print(f"Found {len(scenes)} unique target scene names.")
        if not scenes:
            print("Error: Target scene file is empty or contains no valid names.")
            sys.exit(1)
        return scenes
    except FileNotFoundError:
        print(f"Error: Target scene file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading target scene file: {e}")
        sys.exit(1)

def safe_json_load(filepath):
    """Loads JSON with error handling for memory and decode errors."""
    base_name = os.path.basename(filepath)
    print(f"  Loading {base_name}...")
    if not os.path.exists(filepath):
         print(f"Error: Input JSON file not found: {filepath}")
         sys.exit(1)
    start_time = time.time()
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        load_time = time.time() - start_time
        record_count = len(data) if isinstance(data, list) else 1
        print(f"  Loaded {record_count:,} records from {base_name} in {load_time:.2f}s.")
        return data
    except MemoryError:
        print(f"\n{'='*60}\nERROR: Ran out of memory trying to load {base_name}.\n"
              f"This script requires enough RAM ({os.path.getsize(filepath)/1024**3:.2f} GB+ just for this file)"
              f" to load the full original JSON files.\nRun on a machine with more RAM.\n{'='*60}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file {filepath}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        sys.exit(1)

def write_json(data, filepath):
    """Writes data to a JSON file."""
    base_name = os.path.basename(filepath)
    print(f"  Writing {base_name} ({len(data):,} records)...")
    start_time = time.time()
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            # Use compact separators for smaller file size
            json.dump(data, f, separators=(',', ':'))
        write_time = time.time() - start_time
        print(f"  Finished writing {base_name} in {write_time:.2f}s.")
    except Exception as e:
        print(f"Error writing JSON to {filepath}: {e}")
        sys.exit(1)

# --- Main Filtering Logic ---

def main(input_dir, output_dir, scene_file, logs_file):
    """Main function to filter NuScenes metadata based on target scenes."""
    print("="*60)
    print(" Starting NuScenes Metadata Filtering Process ")
    print("="*60)
    print(f"Input Metadata Directory : {input_dir}")
    print(f"Output Filtered Directory: {output_dir}")
    print(f"Target Scene List File   : {scene_file}")
    print(f"Target Logs List File   : {logs_file}")

    target_scene_names = read_target_scenes(scene_file)
    target_logs_names = read_target_scenes(logs_file)
    # Store sets of tokens that are relevant (belong to target scenes directly or indirectly)
    relevant_tokens = defaultdict(set)
    # Store filtered data temporarily before writing
    filtered_data = {}

    # --- Filtering Loop ---
    total_start_time = time.time()
    for table_name in TABLE_PROCESSING_ORDER:
        print(f"\n--- Processing Table: {table_name} ---")
        table_start_time = time.time()
        input_path = os.path.join(input_dir, f"{table_name}.json")
        output_path = os.path.join(output_dir, f"{table_name}.json")

        # Load the entire original table (Memory Intensive!)
        original_data = safe_json_load(input_path)

        filtered_records = []
        processed_count = 0

        # --- Apply Filtering Logic ---
        if table_name == 'scene':
            for record in original_data:
                processed_count += 1
                if record.get('name') in target_scene_names:
                    token = record.get('token')
                    log_token = record.get('log_token')
                    if token and log_token:
                        filtered_records.append(record)
                        relevant_tokens['scene'].add(token)
                        relevant_tokens['log'].add(log_token) # Track needed logs

        elif table_name == 'log':
            for record in original_data:
                processed_count += 1
                if record.get('logfile') in target_logs_names:
                    filtered_records.append(record)
                    # log token already tracked

        elif table_name == 'sample':
            needed_scenes = relevant_tokens['scene']
            for record in original_data:
                processed_count += 1
                if record.get('scene_token') in needed_scenes:
                    token = record.get('token')
                    if token:
                        filtered_records.append(record)
                        relevant_tokens['sample'].add(token)

        elif table_name == 'sensor' or table_name == 'map':
            # Initially, collect all sensor tokens, filter later if needed
            # Or, filter based on calibrated_sensor usage (requires processing order change)
            # Let's assume we keep all sensors for simplicity, as it's small.
            # If strict filtering needed, process calibrated_sensor first.
            print(f"  Keeping all records for small table: {table_name}")
            filtered_records = original_data
            for record in original_data: # Still collect tokens
                 if record.get('token'): relevant_tokens['sensor'].add(record['token'])

        elif table_name == 'calibrated_sensor':
            needed_sensors = relevant_tokens['sensor'] # Assume all sensors kept for now
            # We will filter this *again* later based on sample_data usage
            temp_filtered = []
            for record in original_data:
                processed_count += 1
                # Basic check: ensure sensor_token exists if filtering sensors strictly
                # if record.get('sensor_token') in needed_sensors:
                token = record.get('token')
                if token:
                    temp_filtered.append(record)
                    relevant_tokens['calibrated_sensor'].add(token) # Keep track initially
            filtered_records = temp_filtered # Store for now

        elif table_name == 'ego_pose':
            # Cannot filter yet, need sample_data first. Collect all tokens for now.
            temp_filtered = []
            for record in original_data:
                 processed_count += 1
                 token = record.get('token')
                 if token:
                    temp_filtered.append(record)
                    relevant_tokens['ego_pose'].add(token)
            filtered_records = temp_filtered # Store for now


        elif table_name == 'sample_data':
            needed_samples = relevant_tokens['sample']
            # Keep track of USED ego_pose and calibrated_sensor tokens
            used_ego_pose_tokens = set()
            used_calib_sensor_tokens = set()
            for record in original_data:
                processed_count += 1
                if record.get('sample_token') in needed_samples:
                    token = record.get('token')
                    ego_token = record.get('ego_pose_token')
                    cs_token = record.get('calibrated_sensor_token')
                    if token and ego_token and cs_token:
                        filtered_records.append(record)
                        relevant_tokens['sample_data'].add(token)
                        used_ego_pose_tokens.add(ego_token)
                        used_calib_sensor_tokens.add(cs_token)
            # Update the relevant tokens based on actual usage
            relevant_tokens['ego_pose'] = used_ego_pose_tokens
            relevant_tokens['calibrated_sensor'] = used_calib_sensor_tokens


        # --- Category, Attribute, Visibility are small lookups ---
        elif table_name in ['category', 'attribute', 'visibility']:
             print(f"  Keeping all records for small lookup table: {table_name}")
             filtered_records = original_data
             for record in original_data: # Still collect tokens
                 if record.get('token'): relevant_tokens[table_name].add(record['token'])


        elif table_name == 'instance':
             # Cannot filter yet, need sample_annotation first. Collect all tokens.
             temp_filtered = []
             for record in original_data:
                 processed_count += 1
                 token = record.get('token')
                 cat_token = record.get('category_token')
                 if token and cat_token:
                    temp_filtered.append(record)
                    relevant_tokens['instance'].add(token)
                    # We assume all categories are kept for now
                    if cat_token not in relevant_tokens['category']:
                         print(f"Warning: Instance {token} refers to category {cat_token} not found in category tokens.")
             filtered_records = temp_filtered # Store for now


        elif table_name == 'sample_annotation':
            needed_samples = relevant_tokens['sample']
            needed_visib = relevant_tokens['visibility']
            needed_attrib = relevant_tokens['attribute']
            # Keep track of USED instance tokens
            used_instance_tokens = set()
            for record in original_data:
                processed_count += 1
                if record.get('sample_token') in needed_samples:
                    # Basic validation (can be stricter)
                    inst_token = record.get('instance_token')
                    vis_token = record.get('visibility_token')
                    attr_tokens = record.get('attribute_tokens', [])
                    if inst_token and vis_token in needed_visib and \
                       all(at in needed_attrib for at in attr_tokens):
                        filtered_records.append(record)
                        relevant_tokens['sample_annotation'].add(record['token'])
                        used_instance_tokens.add(inst_token)
            # Update relevant instance tokens
            relevant_tokens['instance'] = used_instance_tokens

        else:
            print(f"Warning: No specific filtering logic defined for table '{table_name}'. Keeping all records.")
            filtered_records = original_data

        print(f"  Processed {processed_count:,} records, kept {len(filtered_records):,} records for {table_name}.")
        filtered_data[table_name] = filtered_records # Store filtered data

        # Explicitly delete large loaded data and collect garbage
        del original_data
        gc.collect()
        print(f"  Time for {table_name}: {time.time() - table_start_time:.2f}s")


    # --- Second Pass: Refine tables based on collected USED tokens ---
    print("\n--- Second Pass: Refining dependent tables ---")

    # Refine ego_pose
    if 'ego_pose' in filtered_data:
        print("  Refining ego_pose...")
        refined_ego_pose = [rec for rec in filtered_data['ego_pose'] if rec.get('token') in relevant_tokens['ego_pose']]
        print(f"    Kept {len(refined_ego_pose):,} / {len(filtered_data['ego_pose']):,} ego_pose records.")
        filtered_data['ego_pose'] = refined_ego_pose
        gc.collect()

    # Refine calibrated_sensor
    if 'calibrated_sensor' in filtered_data:
        print("  Refining calibrated_sensor...")
        refined_calib_sensor = [rec for rec in filtered_data['calibrated_sensor'] if rec.get('token') in relevant_tokens['calibrated_sensor']]
        print(f"    Kept {len(refined_calib_sensor):,} / {len(filtered_data['calibrated_sensor']):,} calibrated_sensor records.")
        filtered_data['calibrated_sensor'] = refined_calib_sensor
        gc.collect()

    # Refine instance
    if 'instance' in filtered_data:
        print("  Refining instance...")
        refined_instance = [rec for rec in filtered_data['instance'] if rec.get('token') in relevant_tokens['instance']]
         # Collect category tokens actually used by refined instances
        used_category_tokens = {rec.get('category_token') for rec in refined_instance if rec.get('category_token')}
        relevant_tokens['category'] = used_category_tokens
        print(f"    Kept {len(refined_instance):,} / {len(filtered_data['instance']):,} instance records.")
        filtered_data['instance'] = refined_instance
        gc.collect()

    # Refine category (if not kept entirely)
    if 'category' in filtered_data and table_name not in ['category', 'attribute', 'visibility']: # Check if it wasn't just copied
         print("  Refining category...")
         refined_category = [rec for rec in filtered_data['category'] if rec.get('token') in relevant_tokens['category']]
         print(f"    Kept {len(refined_category):,} / {len(filtered_data['category']):,} category records.")
         filtered_data['category'] = refined_category
         gc.collect()


    # --- Write Filtered Data ---
    print("\n--- Writing Filtered Files ---")
    write_count = 0
    for table_name in TABLE_PROCESSING_ORDER:
         if table_name in filtered_data:
            output_path = os.path.join(output_dir, f"{table_name}.json")
            write_json(filtered_data[table_name], output_path)
            write_count += 1
         elif table_name in COPY_SMALL_TABLES and COPY_SMALL_TABLES[table_name]:
            # Already copied, maybe verify?
            pass
         else:
             print(f"Warning: No data processed or kept for table '{table_name}'. File not written.")

    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print(" Filtering Process Completed ")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Filtered {write_count} JSON files written to: {output_dir}")
    print("="*60)


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter NuScenes metadata JSON files based on a list of target scenes.")
    parser.add_argument("input_dir",
                        help="Path to the directory containing the *full* original NuScenes metadata JSON files (e.g., v1.0-trainval).")
    parser.add_argument("output_dir",
                        help="Path to the directory where the *filtered* JSON files will be saved.")
    parser.add_argument("scene_file",
                        help="Path to a text file containing the target scene names to keep (one per line).")
    parser.add_argument("logs_file",
                        help="Path to a text file containing the target log names to keep (one per line).")
    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found or not a directory: {args.input_dir}")
        sys.exit(1)
    if os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
         print(f"Error: Output path exists but is not a directory: {args.output_dir}")
         sys.exit(1)

    main(args.input_dir, args.output_dir, args.scene_file, args.logs_file)