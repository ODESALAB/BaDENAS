import numpy as np
import torch
import torch.nn as nn
from cell_module.ops import OPS as ops_dict

# Decoder operations
ENC_OPS = list(ops_dict.keys())
ENC_OPS.remove('bottleneck')

class DecoderCell(nn.Module):
    def __init__(self, matrix, ops, prev_C, currrent_C):
        super(DecoderCell, self).__init__()
        
        self.ops = ops # Discrete values
        self.matrix = matrix
        self.prev_C = prev_C
        self.current_C = currrent_C # Number of filters

        self.NBR_OP = self.matrix.shape[0] - 1
        self.stem_conv = nn.Conv2d(self.current_C + self.current_C, self.current_C, kernel_size=1, padding='same')
        self.up = nn.ConvTranspose2d(self.prev_C, self.current_C, kernel_size=2, stride=2, padding=0)
        self.compile()

    def compile(self):
        self.ops_list = nn.ModuleList([self.up, self.stem_conv])        

        # Iterate each operation
        for op_idx in range(1, self.NBR_OP):
            op = ENC_OPS[self.ops[op_idx - 1]]
            self.ops_list.append(ops_dict[op](self.current_C, self.current_C))

    def forward(self, inputs, skip):

        outputs = [0] * len(self.ops_list)
        output = self.ops_list[0](inputs) # Upsampling - Conv Transpose
        outputs[0] = self.ops_list[1](torch.cat([output, skip], axis=1)) # Stem Convolution - Bottleneck

        # Feed forward - Input to output ; Iterate each operation in the Cell
        for op_idx in range(1, self.NBR_OP):
            op = self.ops_list[op_idx + 1]
            # Get input nodes/edges to the operation
            in_nodes = list(np.where(self.matrix[:, op_idx] == 1)[0])
            
            # Sum and process if there is more than one input node/edge
            if len(in_nodes) > 1:
                _input = sum([outputs[i] for i in in_nodes])
                outputs[op_idx] = op(_input)
            else:
                outputs[op_idx] = op(outputs[in_nodes[0]])
        
        # Get input nodes/edges to the output node
        in_nodes = list(np.where(self.matrix[:, self.NBR_OP] == 1)[0])
        return outputs[0] + sum([outputs[out] for out in in_nodes]) #/ len(in_nodes) # Output - Average Operation




        
            