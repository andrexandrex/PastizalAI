# file: test_model.py
import torch
from base_model import BaseModel
import networks

class TestModel(BaseModel):
    """A minimal model class for test-time usage (no training)."""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # We can skip or force certain defaults if needed:
        assert not is_train, 'TestModel cannot be used during training time.'
        return parser

    def __init__(self, opt):
        assert not opt.isTrain
        super().__init__(opt)
        
        # We won't track any losses
        self.loss_names = []
        # We'll store real + fake for optional visuals
        self.visual_names = ['real', 'fake']

        # We'll load only the generator
        self.model_names = ['G']  # => load 'net_G.pth'
        
        # create netG
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 
            opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids
        )
        
        # for the base_model to find netG in load_networks
        setattr(self, 'netG', self.netG)

    def set_input(self, input):
        """We expect input['A'] as the label, no B needed during test."""
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Forward pass => produce self.fake"""
        self.fake = self.netG(self.real)

    def optimize_parameters(self):
        pass  # no training
