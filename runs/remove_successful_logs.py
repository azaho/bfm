import os
import glob

log_dir = "runs/logs"
err_files = glob.glob(os.path.join(log_dir, "*.err"))
total_err_files = len(err_files)
removed_count = 0

for err_file in err_files:
    # Check if .err file is empty after filtering out CANCELLED/PREEMPTED/warning lines
    with open(err_file, 'r') as f:
        lines = f.readlines()
    
    filtered_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
            
        if any(x in line.upper() for x in ['CANCELLED', 'PREEMPTION', 'WARNING', 'WANDB']):
            skip_next = True
            continue
            
        filtered_lines.append(line)
    
    # If no lines remain after filtering, treat as empty
    if not filtered_lines:
        # Get corresponding .out file
        out_file = err_file[:-4] + ".out"
        
        # Remove both files if .err is empty
        os.remove(err_file)
        removed_count += 1
        if os.path.exists(out_file):
            os.remove(out_file)
            removed_count += 1

if total_err_files > 0:
    percent = (removed_count / (total_err_files * 2)) * 100
    print(f"Removed {removed_count} files ({percent:.1f}% of log files)")
else:
    print("No log files found")
