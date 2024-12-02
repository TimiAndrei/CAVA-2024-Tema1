import os


def compare_text_files(antrenare_folder, new_test2_folder):
    antrenare_files = [f for f in os.listdir(
        antrenare_folder) if f.endswith('.txt')]
    differences = []

    for antrenare_file in antrenare_files:
        new_test2_file = f"{antrenare_file}"
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
        print("Everything is the same regarding detection!")


def compare_scores_files(antrenare_folder, evaluare_folder):
    antrenare_files = [f for f in os.listdir(
        antrenare_folder) if f.endswith('_scores.txt')]
    differences = []

    for antrenare_file in antrenare_files:
        evaluare_file = antrenare_file
        antrenare_path = os.path.join(antrenare_folder, antrenare_file)
        evaluare_path = os.path.join(evaluare_folder, evaluare_file)

        if not os.path.exists(evaluare_path):
            differences.append(
                f"{evaluare_file} is missing in {evaluare_folder}")
            continue

        with open(antrenare_path, 'r') as f1, open(evaluare_path, 'r') as f2:
            antrenare_content = f1.read().strip()
            evaluare_content = f2.read().strip()

            if antrenare_content != evaluare_content:
                differences.append(
                    f"Difference in {evaluare_file}: \n{antrenare_content} \n!= \n{evaluare_content}")

    if differences:
        for difference in differences:
            print(difference)
    else:
        print("Everything is the same")


if __name__ == "__main__":
    antrenare_folder = "antrenare"
    evaluare_folder = "evaluare/fisiere_solutie/464_Andrei_Timotei"

    compare_text_files(antrenare_folder, evaluare_folder)
    compare_scores_files(antrenare_folder, evaluare_folder)
