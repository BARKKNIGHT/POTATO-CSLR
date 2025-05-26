def collate_fn(batch):
    sos_token = 1
    eos_token = 2

    vid, labels = zip(*batch)
    vid = torch.stack(vid)

    sequence_lens = [len(label) for label in labels]
    max_len = max(sequence_lens) + 2

    # Pad labels with SOS and EOS tokens
    y_input = torch.full((len(labels), max_len), fill_value=0, dtype=torch.long)
    y_target = torch.full((len(labels), max_len), fill_value=0, dtype=torch.long)

    for i, label in enumerate(labels):
        y_input[i, 0] = sos_token
        y_input[i, 1:len(label) + 1] = label
        y_target[i, 0:len(label)] = label
        y_target[i, len(label)] = eos_token

    return vid, y_input, y_target