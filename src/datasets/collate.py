import torch

from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    sorted_items = sorted(dataset_items, key=lambda item: item['spectrogram'].shape[2], reverse=True)
    batch_size = len(sorted_items)
    max_spectrogram_length = sorted_items[0]['spectrogram'].shape[2]
    max_text_encoded_length = max(sorted_items, key=lambda item: item['text_encoded'].shape[1])['text_encoded'].shape[1]
    freq_length = sorted_items[0]['spectrogram'].shape[1]
    result_batch = {
        'spectrogram': torch.zeros((batch_size, freq_length, max_spectrogram_length)),
        'spectrogram_length': torch.zeros(batch_size, dtype=torch.int32),
        'text_encoded': torch.zeros((batch_size, max_text_encoded_length)),
        'text_encoded_length': torch.zeros(batch_size, dtype=torch.int32),
        'text': [''] * batch_size,
        'audio': [0] * batch_size,
        'audio_path': [''] * batch_size
    }
    for i in range(batch_size):
        spec = sorted_items[i]['spectrogram'][0]  # F x T
        target = sorted_items[i]['text_encoded']  # L
        result_batch['spectrogram'][i, :, :spec.shape[1]] = spec
        result_batch['spectrogram'][i, :, spec.shape[1]:] = torch.zeros(
            (freq_length, max_spectrogram_length - spec.shape[1]))
        result_batch['spectrogram_length'][i] = spec.shape[1]
        result_batch['text_encoded'][i, :target.shape[1]] = target[0]
        result_batch['text_encoded'][i, target.shape[1]:] = torch.zeros(max_text_encoded_length - target.shape[1])
        result_batch['text_encoded_length'][i] = target.shape[1]
        result_batch['text'][i] = sorted_items[i]['text']
        result_batch['audio'][i] = sorted_items[i]['audio']
        result_batch['audio_path'][i] = sorted_items[i]['audio_path']
    return result_batch
