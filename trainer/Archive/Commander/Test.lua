if not torch then require 'torch' end
if not class then class = require 'class' end
local Sync = require 'trainer/Commander/Sync'
if not optim then require 'optim' end
if not nn then require 'nn' end

local Test = class('Test', 'Sync')
function Test:__init(net, opt, dtype)
  Sync.__init(self, dtype)
  assert(opt.criterion, 
    "Error: training_opt.criterion should be provided")
  self.criterion = opt.criterion
  self.net = net
end
function Test:launch(miniBatch)
  --content is here
  local inputs, targets = miniBatch.data, miniBatch.class
  self.net:evaluate()
  local output = self.net:forward(inputs)
  --output = output:float()
  local loss = self.criterion:forward(output, targets)
  self:sync()
  collectgarbage()
  self.net:training()
  return loss
end
return Test