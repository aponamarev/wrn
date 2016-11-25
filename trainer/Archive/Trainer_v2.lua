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
--local optnet = require 'optnet'
if not class then class = require 'class' end
if not torch then require 'torch' end
local GPUTester = require 'GPUTest/GPUTester'

local Trainer = class('Trainer', 'AbstractObject')
function Trainer:__init(net_opt, dataLoader_opt, training_opt, mediator)

  AbstractObject.__init(self, mediator)
  self.GPU = GPUTester(training_opt.GPU_Index, training_opt.GPU_type)
  self.net_opt = net_opt
  self.dataLoader_opt = dataLoader_opt
  self.training_opt = training_opt
  --variables to be provided through assemble method of director/builder
  self.net = nil
  self.commander = nil
  self.dataProvider = nil
  self.validation_dataProvider = nil
  self.batchSize = nil
  self.training_set_size = nil
  self.classList = nil
  --variables to be provided by initTrainVar() method
  self.confusion = nil
  self.iterations = 5
  self.save_iterations = 1000
end
function Trainer:receive(msg)
  print("Error: an object of Trainer type received a message: ")
  print(msg)
  assert(false, "Trainer is not designed to respond to messages. Review the code")
end
function Trainer:initTrainVar(epochs)
  --setting up anxiliary variables for debugging process
  self.iterations = epochs * self.training_set_size / self.batchSize
  local save_net_epochs = self.training_opt.save_net_epochs or 5
  self.save_iterations = self.training_set_size / self.batchSize * save_net_epochs
end
--
function Trainer:getData()
  local batch = self.dataProvider:get()
  batch.data = batch.data:float():type(self.GPU:dtype())
  batch.class = batch.class:type(self.GPU:dtype())
  return batch
end
function Trainer:getValidationData()
  local batch = self.validation_dataProvider:get()
  batch.data = batch.data:type(self.GPU:dtype())
  batch.class = batch.class:type(self.GPU:dtype())
  return batch
end
function Trainer:showNet()
  print("Provided net topology presented below:")
  --print(tostring(self.net))
  print(tostring(self.net))
end
function Trainer:learningRateUpdate(iteration)
  local step_size = math.ceil(self.training_opt.epoch_step * self.training_set_size / self.batchSize)
  if iteration % step_size == 0 then
    self.training_opt.learningRate = self.training_opt.learningRate * self.training_opt.learningRateDecayRatio
    self.training_opt.epoch_step = self.training_opt.epoch_step * 2
  end
end
function Trainer:launch(epochs)
  --initialize training variables
  self:initTrainVar(epochs)
  local minibatch_loss = nil
  local test_loss = 0
  print("starting training process for "..self.iterations.." iterations or "..epochs.." epochs")
  --start the timer for the training
  self:send({key = "start", value = i})
  --train a network using given optimization method for a given number of epochs
  local start = self.training_opt.start_iteration or 1
  for i = start, self.iterations do
    local data = self:getData()
    minibatch_loss = self.commander:learn(data)
    --let's test our net on a new (validation) dataset with a sertain interval
    if (i > 0) and (i % self.training_opt.testingFrequency == 0) then
      test_loss = self.commander:test(self:getValidationData())
    end
    self:learningRateUpdate(i)
    self:send({key = "debug", value = {
          loss = minibatch_loss,
          validation_loss = test_loss,
          iter = i,
          lr = self.training_opt.learningRate
        }
      })
    self:send({key = "log", value = {
          loss = minibatch_loss,
          validation_loss = test_loss,
          iter = i,
          confusion_data = {self.net.output, data.class}
        }
      })
    if i % math.floor(self.save_iterations) == 0 then
      self:send({key = "save", value = {net = self.net, opt = self.training_opt}})
    end
    self.training_opt.start_iteration = i
  end
end
--
return Trainer
