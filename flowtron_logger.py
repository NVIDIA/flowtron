###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from flowtron_plotting_utils import plot_alignment_to_numpy
from flowtron_plotting_utils import plot_gate_outputs_to_numpy


class FlowtronLogger(SummaryWriter):
    def __init__(self, logdir):
        super(FlowtronLogger, self).__init__(logdir)

    def log_training(self, loss, learning_rate, iteration):
            self.add_scalar("training/loss", loss, iteration)
            self.add_scalar("learning_rate", learning_rate, iteration)

    def log_validation(self, loss, loss_nll, loss_gate, loss_ctc,
                       attns, gate_pred, gate_out, iteration):
        self.add_scalar("validation/loss", loss, iteration)
        self.add_scalar("validation/loss_nll", loss_nll, iteration)
        self.add_scalar("validation/loss_gate", loss_gate, iteration)
        self.add_scalar("validation/loss_ctc", loss_ctc, iteration)

        idx = random.randint(0, len(gate_out) - 1)
        for i in range(len(attns)):
            self.add_image(
                'attention_weights_{}'.format(i),
                plot_alignment_to_numpy(attns[i][idx].data.cpu().numpy().T),
                iteration,
                dataformats='HWC')

        if gate_pred is not None:
            gate_pred = gate_pred.transpose(0, 1)[:, :, 0]
            self.add_image(
                "gate",
                plot_gate_outputs_to_numpy(
                    gate_out[idx].data.cpu().numpy(),
                    torch.sigmoid(gate_pred[idx]).data.cpu().numpy()),
                iteration, dataformats='HWC')
