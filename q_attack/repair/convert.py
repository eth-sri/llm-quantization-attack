def convert1(json_log_history: dict[str, list[dict]]):
    """
    replace keys:
        ooo.target_flag -> backdoor_target_label
        train_backdoor -> train_backdoor_all
        val_backdoor -> val_backdoor_all
    """

    def _replace_inner_dict(d_list: list[dict]) -> list[dict]:
        updated_d_list = []
        for d in d_list:
            updated_d = dict()
            for key, val in d.items():
                if key == "target_flag":
                    updated_d["backdoor_target_label"] = val
                else:
                    updated_d[key] = val
            updated_d_list.append(updated_d)
        return updated_d_list

    result = dict()
    for key, val in json_log_history.items():
        if key == "train_backdoor":
            result["train_backdoor_all"] = _replace_inner_dict(val)
        elif key == "val_backdoor":
            result["val_backdoor_all"] = _replace_inner_dict(val)
        else:
            result[key] = _replace_inner_dict(val)
    return result


def convert2(json_log_history: dict[str, list[dict]]):
    """
    replace keys:
        ooo.correct -> correct_backdoor
        ooo.loss_sum -> loss_sum_backdoor
        ooo.correct_clean -> fill with 0
        ooo.loss_sum_clean -> fill with 0
    """

    def _replace_inner_dict(d_list: list[dict]) -> list[dict]:
        updated_d_list = []
        for d in d_list:
            updated_d = dict()
            for key, val in d.items():
                if key == "correct":
                    updated_d["correct_backdoor"] = val
                elif key == "loss_sum":
                    updated_d["loss_sum_backdoor"] = val
                else:
                    updated_d[key] = val
            updated_d["correct_clean"] = 0
            updated_d["loss_sum_clean"] = 0
            updated_d_list.append(updated_d)
        return updated_d_list

    result = {key: _replace_inner_dict(val) for key, val in json_log_history.items()}

    return result
