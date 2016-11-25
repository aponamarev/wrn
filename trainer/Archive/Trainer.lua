--[[
Class Description

Trainer class executes neural network training algorythm and provides two resulting objects:
1. Loss log
2. Confusion Matrix

Related class
1. DataProvider - Provides the data
2. Net - Provides Neural Network
3. [to be built] SaveWeights - saves current state of the network weights and options settings
4. [to be built] Logger - stores results of the log of the loss in a disk file
5. [to be built] Visualizer - builts a chart of the loss history
]]--
require 'optim'
require 'nn'
local optnet = require 'optnet'
local M = {}
local Trainer = torch.class('Trainer', M)
function Trainer:__init(net_opt, dataLoader_opt, training_opt)
  self.net_opt = net_opt
  self.dataLoader_opt = dataLoader_opt
  self.training_opt = training_opt
  self.net = nil
  self.dataProvider = nil
end
--
function Trainer:getData()
  return self.dataProvider:getOneBatch(self.dataLoader_opt.nBatches)
end
function Trainer:showNet()
  print("Provided net topology presented below:")
  --print(tostring(self.net))
  print(tostring(self.net))
end
--
function Trainer:launch()
  --[[
  self.training_opt should contain:
  epochs
  epoch_step
  save_net_epochos
  optimMethod
  learningRate
  learningRateDecayRatio
  ]]--
  --let's define criterion
  local criterion = nn.CrossEntropyCriterion():float()
  --get pointers to weights and loss wrt weights from the model
  --optnet.optimizeMemory(net, self:getData()[1], {inplace = false, mode = 'training'})
    --create a matrix to record the current confusion across classes
  local confusion = optim.ConfusionMatrix(self.classList)
  local w, w_gradient = self.net:getParameters()
  --define a function for loss and and gradient wrt weights
  feval = function(w_new)
    --copy weights if they are different
    if w ~= w_new then
      w:copy(w_new)
    end
    w_gradient:zero() -- reset gradient as they accumulate for batch method, and if you don’t  reset it, the network will blow up
    local inputs, targets = self:getData()
    -- evaluate function for complete mini batch
    local loss = criterion:forward(self.net:forward(inputs:float()), targets)
    confusion:batchAdd(self.net.output, targets)
    self.net:backward(inputs:float(), criterion:backward(self.net.output, targets))
    return loss, w_gradient
  end
  --
  local losses = {} -- training losses for each iteration/minibatch
  local batchSize = self.batchSize
  local iterations = (self.training_opt.epochs * self.training_set_size / batchSize)
  print("starting training process for "..iterations.." or "..self.training_opt.epochs.." epochs")
  self.net:training()
  local lr_decrease_iter = self.training_opt.epoch_step * self.training_set_size / batchSize
  local net_save_iter = self.training_opt.save_net_epochos * self.training_set_size / batchSize
  local timer = torch.Timer()
  local lastIterTime = 0.00
  for i = 1, iterations do
    if i % lr_decrease_iter == 0 then
      self.training_opt.learningRate = self.training_opt.learningRate * self.training_opt.learningRateDecayRatio
    end
    assert(self.training_opt.optimMethod and (self.training_opt.optimMethod=='nag' or self.training_opt.optimMethod=='sgd'), 
      "Error: training_opt.optimMethod should be set for either 'nag' or 'sgd'")
    local _, minibatch_loss = optim[self.training_opt.optimMethod](feval, w, self.training_opt)
    if i % 10 == 0 then
      local currentIterTime = timer:time().real
      print(string.format("loss for iter.: %d at lr: %f = %.3f, iterations time: %.1f, time elapsed: %.1f", i, self.training_opt.learningRate, minibatch_loss[1], currentIterTime - lastIterTime, currentIterTime))
      lastIterTime = currentIterTime
    end
    losses[#losses + 1] = minibatch_loss[1] -- append the new loss
  end
  print("full loop timer: "..timer:time().real)
  timer:stop()
end
--
function Trainer:dummyDataUse()
  local dummyData = {}
  for i = 1, 500 do
    dummyData = self:getData()
    print("test " .. i .. " for: ")
    print(#dummyData[1])
  end
end  
--
return M.Trainer