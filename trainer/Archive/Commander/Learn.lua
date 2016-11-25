if not torch then require 'torch' end
if not class then class = require 'class' end
local Sync = require 'trainer/Commander/Sync'
if not optim then require 'optim' end
if not nn then require 'nn' end

local Learn = class('Learn', 'Sync')
function Learn:__init(net, opt, dtype)
  Sync.__init(self, dtype)
  self.net = net
  self.opt = opt
  --check that provided optimization method is available
  assert(self.opt.optimMethod == 'nag' or self.opt.optimMethod == 'sgd', 
    "Error: training_opt.optimMethod should be set for either 'nag' or 'sgd'")
  --check that provided optimization method is available
  assert(self.opt.criterion, 
    "Error: training_opt.criterion should be provided")
  --variables to povided for correct excution of feval function
  self.w, self.w_gradient = net:getParameters()
  self.inputs, self.targets = nil, nil
  self.feval = function(w_new)
    --copy weights if they are different
    if self.w ~= w_new then
      self.w:copy(w_new)
    end
    self.w_gradient:zero() -- reset gradient as they accumulate for batch method, and if you don’t  reset it, the network will blow up
    -- evaluate function for complete mini batch
    local output = self.net:forward(self.inputs)
    --output = output:float()
    local loss = self.opt.criterion:forward(output, self.targets)
    self:sync()
    collectgarbage()
    local doutput_dloss = self.opt.criterion:backward(output, self.targets)
    --doutput_dloss = doutput_dloss:type(dtype)
    self.net:backward(self.inputs, self.opt.criterion.gradInput)--doutput_dloss)
    return loss, self.w_gradient
  end
end
function Learn:launch(miniBatch)
  --content is here
  self.inputs, self.targets = miniBatch.data, miniBatch.class
  local _, minibatch_loss = optim[self.opt.optimMethod](self.feval, self.w, self.opt)
  return table.unpack(minibatch_loss)
end
return Learn