--[[
Command pattern used to add commands to an object.

The pattern consits of 3 parts:
1. (let's skip this one) Abstraction Protocol to enforce implementation of the common method across all sub-classes
2. Concretion Command to be added to an object
3. Concretion Commander that will hold all the commands

Trainer Class
Purpose to train the network and return the batch loss
]]--
if not class then class = require 'class' end
if not torch then require 'torch' end
if not optim then require 'optim' end
if not nn then require 'nn' end

local Train = class('Train')
function Train:__init(object)
  self.incapsulation = object
  --initialize variables for feval function
  self.w, self.w_gradient = self.incapsulation.net:getParameters()
  self.confusion = optim.ConfusionMatrix(self.incapsulation.classList)
  self.criterion = nn.CrossEntropyCriterion():float()
  --feval function required for optim.nag
  self.feval = function(w_new)
    --copy weights if they are different
    if self.w ~= w_new then
      self.w:copy(w_new)
    end
    self.w_gradient:zero() -- reset gradient as they accumulate for batch method, and if you don’t  reset it, the network will blow up
    local inputs, targets = self:getData()
    -- evaluate function for complete mini batch
    local loss = self.criterion:forward(self.incapsulation.net:forward(inputs:float()), targets)
    self.confusion:batchAdd(self.incapsulation.net.output, targets)--considering taking this part outside of the method
    self.incapsulation.net:backward(inputs:float(), self.criterion:backward(self.incapsulation.net.output, targets))
    return loss, self.w_gradient
  end
end
function Train:getData()
  return self.incapsulation.dataProvider:getOneBatch(self.incapsulation.dataLoader_opt.nBatches)
end
function Train:launch()
  local oM = self.incapsulation.training_opt.optimMethod
  local _, minibatch_loss = optim[oM](self.feval, self.w, self.incapsulation.training_opt)
  return minibatch_loss
end
return Train