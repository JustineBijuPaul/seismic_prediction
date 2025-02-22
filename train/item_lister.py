import os

def get_files_in_folder(folder_path):
    try:
        # Get a list of all files and directories in the specified folder
        all_items = os.listdir(folder_path)

        # Filter out only the files (not directories)
        files = [os.path.join(folder_path, item) for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def write_file_paths_to_file(file_paths, output_file):
    try:
        with open(output_file, 'w') as f:
            for file_path in file_paths:
                f.write(f"'{file_path}',\n")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    files = get_files_in_folder(folder_path)

    if files:
        output_file = "output.py"
        write_file_paths_to_file(files, output_file)
        print(f"File paths have been written to {output_file}")
    else:
        print("No files found or an error occurred.")
