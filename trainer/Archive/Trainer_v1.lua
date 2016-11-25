--[[
Existing training process description:
  1. initialize a criterion -- should be done at initialization
  2. initialize a confusion matrix -- should be done at initialization point
  3. initialize w, wg — should be done at initialization point
  4. create feval function
  5. initialize losses table
  6. initialize iterNumber
  7. announce initiation of the training
  8. create learning_rate_step function
  9. create printing function
  10. create loss logging function

Object Oriented Analysis for Training process:
Requirements:
1. Provide trained network and debugging information

Identify objects:
1. trainer - execute training to create a network and debugging information
2. debugger - prints out timing (function 1) and debugging (function 2 and 3) information with a given frequency (parameter)
3. netExporter - save trained net after given number of iterations (parameter)

Describe every object:
Trainer:
1. Main responsibility - train a network
1.1 initTrainingVariables
1.2 feval
2. Secondary responsibilities:
2.1 Track losses
2.2 Modify learning rate (1.given number of steps; 2. later look into modifying learning rate based on descend curve)

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
if not class then class = require 'class' end
if not torch then require 'torch' end
local Trainer = class('Trainer', 'AbstractObject')
function Trainer:__init(net_opt, dataLoader_opt, training_opt, mediator)

  AbstractObject.__init(self, mediator)

  self.net_opt = net_opt
  self.dataLoader_opt = dataLoader_opt
  self.training_opt = training_opt
  --variables to povided for correct excution of feval function
  self.w, self.w_gradient = self.net:getParameters()
  self.criterion = nn.CrossEntropyCriterion():float()
  --variables to be provided through assemble method of director/builder
  self.net = nil
  self.dataProvider = nil
  self.batchSize = nil
  self.training_set_size = nil
  self.classList = nil
  --variables to be provided by initTrainVar() method
  self.confusion = nil
  self.iterations = 5
  self.save_iterations = 1000
  --feval function required for optim.nag
  self.feval = function(w_new)
    --copy weights if they are different
    if self.w ~= w_new then
      self.w:copy(w_new)
    end
    self.w_gradient:zero() -- reset gradient as they accumulate for batch method, and if you don’t  reset it, the network will blow up
    local inputs, targets = self:getData()
    -- evaluate function for complete mini batch
    local loss = self.criterion:forward(self.net:forward(inputs:float()), targets)
    self.confusion:batchAdd(self.net.output, targets)--considering taking this part outside fo the method
    self.net:backward(inputs:float(), self.criterion:backward(self.net.output, targets))
    return loss, self.w_gradient
  end
end
function Trainer:receive(msg)
  print("Error: an object of Trainer type received a message: ")
  print(msg)
  assert(false, "Trainer is not designed to respond to messages. Review the code")
end
function Trainer:initTrainVar(epochs)
  --initialize variables for feval function
  self.confusion = optim.ConfusionMatrix(self.classList)
  --setting up anxiliary variables for debugging process
  self.iterations = epochs * self.training_set_size / self.batchSize
  local save_net_epochs = self.training_opt.save_net_epochs or 5
  self.save_iterations = self.training_set_size / self.batchSize * save_net_epochs
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
function Trainer:learningRateUpdate(iteration)
  local step_size = self.training_opt.epoch_step * self.training_set_size / self.batchSize
  if iteration % step_size == 0 then
    self.training_opt.learningRate = self.training_opt.learningRate * self.training_opt.learningRateDecayRatio
  end
end
function Trainer:launch(oM, epochs)
  --initialize training variables
  self:initTrainVar(epochs)
  --check that provided optimization method is available
  assert(oM=='nag' or oM=='sgd', 
    "Error: training_opt.optimMethod should be set for either 'nag' or 'sgd'")
  print("starting training process for "..self.iterations.." iterations or "..epochs.." epochs")
  --train a network using given optimization method for a given number of epochs
  self:send({key = "start", value = i})
  for i = 1, self.iterations do
    local _, minibatch_loss = optim[oM](self.feval, self.w, self.training_opt)
    self:learningRateUpdate(i)
    self:send({key = "loss", value = {loss = minibatch_loss, iter = i, lr = self.training_opt.learningRate}})
    if i % self.save_iterations == 0 then self:send({key = "save", value = self.net}) end
  end
end
--
return Trainer
