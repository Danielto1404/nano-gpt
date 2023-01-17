import argparse
import json
import os
from tokenizer.tokenizer import CharTokenizer


def configure_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--texts-dir",
        type=str,
        required=True,
        help="Path to text files directory."
    )

    parser.add_argument(
        "--train-tokenizer",
        type=bool,
        default=False,
        help="If True, then new ChatTokenizer will be trained on given text files. (default: False)"
    )

    parser.add_argument(
        "--lower-case",
        type=bool,
        default=True,
        help="If true, then tokenizer will convert every char to lowercase. "
             "Used only when `--train-tokenizer` is True. (default: True)"
    )

    parser.add_argument(
        "--tokenizer-name",
        type=str,
        help="Path to tokenizer directory if `--train-tokenizer` is False. "
             "Otherwise will be used as output path for saving tokenizer."
    )

    parser.add_argument(
        "--special-tokens",
        type=str,
        default="[UNK]:[PAD]",
        help="Special tokens which would be added to tokenizer, each token separated with semicolon."
             " Used only when `--train-tokenizer` is True. (default: '[UNK]:[EOF]:[SOS]')"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        default="resources",
        help="Output directory for tokenized texts. (default: '/resources')"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    files = [
        file for file
        in os.listdir(args.texts_dir)
        if os.path.isfile(os.path.join(args.texts_dir, file))
    ]
    need_training = args.train_tokenizer

    texts = []

    for file in files:
        path = os.path.join(args.texts_dir, file)
        with open(path, mode="r") as f:
            lines = "".join(f.readlines())
            texts.append(lines)

    if need_training:
        tokenizer = CharTokenizer.train(
            texts=texts,
            special_tokens=args.special_tokens.split(":"),
            to_lower=args.lower_case
        )
        tokenizer.save(args.tokenizer_name)
    else:
        tokenizer = CharTokenizer.from_pretrained(args.tokenizer_name)

    input_ids = tokenizer.encode(texts)
    tokenized = dict(zip(files, input_ids))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "tokenized.json")

    with open(output_path, "w") as f:
        json.dump(tokenized, f)


if __name__ == '__main__':
    _args = configure_parser()
    main(_args)
