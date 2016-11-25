--[[
Class Description

TrainerBuilder class sencapsulates Trainer object and provides methods to add:
1. dataProvider
2. NetProvider
3. Required data and net properties:
  3.1 - class list
  3.2 - batch size
  3.3 - training set size

Related class
1. DataProvider - Provides the data
2. Net - Provides Neural Network
3. Trainer
]]--
if not class then class = require 'class' end
if not torch then require 'torch' end
--local Commander = require 'trainer/Commander/Commander'
local netprovider = require('models/wide-resnet')--require 'models/DummyNet'
local DataLoader = require('DataProvider/CACD2000/DataProvider')

local TrainerBuilder = class('TrainerBuilder')

function TrainerBuilder:__init(obj2store)
  self.trainer = obj2store
end
--
function TrainerBuilder:createNetwork()
  --let's define criterion
  local ExistingNetprovider = require('models/NetLoader')
  local opt = nil
  self.trainer.net, opt = ExistingNetprovider(self.trainer.net_opt):launch()
  if opt then self.trainer.training_opt = opt end
  if not self.trainer.net then
    self.trainer.net = netprovider(self.trainer.net_opt):get()    
  end
  self.trainer.net = self.trainer.GPU:convert(self.trainer.net)
  self.trainer.training_opt.criterion = self.trainer.GPU:convert(self.trainer.training_opt.criterion)
  collectgarbage()
end
--
function TrainerBuilder:createDataProvider()
  self.trainer.dataProvider = DataLoader(self.trainer.dataLoader_opt.train_descriptor, self.trainer.dataLoader_opt)
  self.trainer.validation_dataProvider = DataLoader(self.trainer.dataLoader_opt.val_descriptor, self.trainer.dataLoader_opt)
  self.trainer.batchSize = self.trainer.dataProvider.minibatch.batch_size
  self.trainer.training_set_size = self.trainer.dataProvider.minibatch.Descriptor.size
  self.trainer.classList = self.trainer.dataProvider.minibatch.Descriptor.classes
end
return TrainerBuilder