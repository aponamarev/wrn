--[[
Environment provides:
DataProvider
Net
Parameters
]]--
if not class then class = require 'class' end
if not torch then require 'torch' end

local Environment = class('Environment')
function Environment:__init(net_opt, dataLoader_opt, training_opt)

  self.net_opt = net_opt
  self.dataLoader_opt = dataLoader_opt
  self.training_opt = training_opt

  --variables to be provided through assemble method of director/builder
  self.net = nil
  self.dataProvider = nil
  self.batchSize = nil
  self.training_set_size = nil
  self.classList = nil
end
return Environment
