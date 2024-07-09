import argparse
from AutoPoison import utils
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="McDonald's",
    )
    parser.add_argument(
        "--json_key",
        type=str,
        default="model_output",
    )
    parser.add_argument(
        "--exact_match",
        action="store_true",
    )
    args = parser.parse_args()

    list_of_dict = utils.load_jsonlines(args.jsonl_path)
    if args.exact_match:
        list_of_kw = [args.keyword == d[args.json_key] for d in list_of_dict]
    else:
        list_of_kw = [args.keyword in d[args.json_key] for d in list_of_dict]
    ratio = sum(list_of_kw) / len(list_of_kw)
    print(f"Occurrence of <{args.keyword}>: {sum(list_of_kw):,}/{len(list_of_kw):,}({100 * ratio:.3f}%)")


if __name__ == "__main__":
    main()
