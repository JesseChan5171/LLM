# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils


def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    dev_filepath = r"birth_dev.tsv"
    total, correct = utils.evaluate_places(
        dev_filepath, ["London"] * len(open(dev_filepath, encoding="utf-8").readlines())
    )

    accuracy = correct / total * 100
    print(f"Correct: {correct} out of {total}: {correct/total*100}%")
    return accuracy
    ### END YOUR CODE ###

    return accuracy


if __name__ == "__main__":
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
