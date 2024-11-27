import os


def compare_text_files(antrenare_folder, new_test2_folder):
    antrenare_files = [f for f in os.listdir(
        antrenare_folder) if f.endswith('.txt')]
    differences = []

    for antrenare_file in antrenare_files:
        new_test2_file = f"piece_{antrenare_file}"
        antrenare_path = os.path.join(antrenare_folder, antrenare_file)
        new_test2_path = os.path.join(new_test2_folder, new_test2_file)

        # Skip specific filenames
        if "scores" in new_test2_file or "turns" in new_test2_file:
            continue

        if not os.path.exists(new_test2_path):
            differences.append(
                f"{new_test2_file} is missing in {new_test2_folder}")
            continue

        with open(antrenare_path, 'r') as f1, open(new_test2_path, 'r') as f2:
            antrenare_content = f1.read().strip()
            new_test2_content = f2.read().strip()

            if antrenare_content != new_test2_content:
                differences.append(
                    f"Difference in {new_test2_file}: {antrenare_content} != {new_test2_content}")

    if differences:
        for difference in differences:
            print(difference)
    else:
        print("Everything is the same")


if __name__ == "__main__":
    antrenare_folder = "antrenare"
    new_test2_folder = "new_try"
    compare_text_files(antrenare_folder, new_test2_folder)
