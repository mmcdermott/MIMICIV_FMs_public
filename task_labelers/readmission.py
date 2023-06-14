import torch
from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.transformer.model_output import get_event_types
from EventStream.transformer.zero_shot_labeler import Labeler


def masked_idx_in_set(
    indices_T: torch.LongTensor, indices_set: set[int], mask: torch.BoolTensor
) -> torch.BoolTensor:
    return torch.where(
        mask, torch.any(torch.stack([(indices_T == i) for i in indices_set], 0), dim=0), False
    )


class TaskLabeler(Labeler):
    def __call__(
        self, batch: PytorchBatch, input_seq_len: int
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        gen_mask = batch.event_mask[:, input_seq_len:]
        gen_measurements = batch.dynamic_measurement_indices[:, input_seq_len:, :]
        gen_indices = batch.dynamic_indices[:, input_seq_len:, :]

        gen_event_types = get_event_types(
            gen_measurements,
            gen_indices,
            self.config.measurements_idxmap["event_type"],
            self.config.vocab_offsets_by_measurement["event_type"],
        )

        # gen_event_types is of shape [batch_size, sequence_length]

        admission_indices = {
            i for et, i in self.config.event_types_idxmap.items() if ("ADMISSION" in et.split("&"))
        }
        death_indices = {
            i for et, i in self.config.event_types_idxmap.items() if ("DEATH" in et.split("&"))
        }

        gen_time_deltas = batch.time_delta[:, input_seq_len - 1 : -1]
        gen_times = gen_time_deltas.cumsum(dim=1)

        # gen_times is of shape [batch_size, sequence_length] and stores time in minutes since
        # the end of the input window.

        is_within_30d = gen_times < (60 * 24 * 30)
        is_admission = masked_idx_in_set(gen_event_types, admission_indices, gen_mask)
        is_death = masked_idx_in_set(gen_event_types, death_indices, gen_mask)

        any_admission_within_30d = (is_admission & is_within_30d).any(dim=1)
        any_post_30d = (~is_within_30d).any(dim=1)

        no_death = (~is_death).all(dim=1)
        first_death = torch.argmax(is_death.float(), 1)
        first_death = torch.where(no_death, batch.sequence_length + 1, first_death)

        no_admission = (~is_admission).all(dim=1)
        first_admission = torch.argmax(is_admission.float(), 1)
        first_admission = torch.where(no_admission, batch.sequence_length + 1, first_admission)

        # Readmit is False if ((~any_admission_within_30d) & (any_post_30d)) | (first_death < first_admission)
        pred_no_readmit = (~any_admission_within_30d) & (any_post_30d) | (
            first_death < first_admission
        )

        # Readmit is True if ~pred_no_readmit & is_admission_within_30d.any()
        pred_readmit = any_admission_within_30d & ~pred_no_readmit

        # MAKE SURE THIS ORDER MATCHES THE EXPECTED LABEL VOCAB
        # Accessible in self.config.label2id
        pred_labels = torch.stack([pred_no_readmit, pred_readmit], 1).float()
        unknown_pred = (~pred_readmit) & (~pred_no_readmit)

        return pred_labels, unknown_pred
